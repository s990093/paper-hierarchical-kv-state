# STATUS_DATA（icpe-data，acvlab）

> 只由 `icpe-data` 寫。格式照 NEXT_PLAN_20261104 §3b：日期時間／做完什麼（commit、run_id）／卡住什麼（完整錯誤）／需要 icpe-sim 做什麼／需要使用者決定什麼。最新的一節在最上面。

## 2026-11-05T07:26+08:00 — G0（merge 7539f01 後）PASS；`cw_main_ttl7` → `tl_main` 開跑

- **G0 PASS**：run `20261105-070841-icpe-data-g0-merged-7539f01`（程式 cc6b980＝7539f01、新功能全關、共用 venv、32 worker、sweep 922 s）：**4,416 列 × 29 欄＝128,064 格，0 個不同**（`G0_RESULT IDENTICAL`）。
- **開跑**：`python -I code/icpe/next_b_queue.py run --plans cw_main_ttl7 tl_main --workers 32`（setsid nohup，程式 111bcf3；log `/ssd6/hungwei/icpe-next/runs/queue-cw_main_ttl7-tl_main-20261105-072434.log`）。先 `cw_main_ttl7` 的 17 個 block，再 TraceLab c1 → c2 → c4。
- 記憶體監看：`/ssd6/hungwei/icpe-next/runs/mem-watch-queue-20261105-072434.log`，每 60 s 記一筆。可用記憶體 <40 GiB 時通知我，我再降 worker。開跑時可用 332 GiB。

## 2026-11-05T07:20+08:00 — `cw_main` 跑完；merge 7539f01、G0 跑中；CacheWise ≥10% 補跑的格子

**做完什麼**
- **`cw_main` 12 個 block 全部 exit 0**（01:50 → 07:07，5.3 h，32 worker；每個 block 22.8–31.6 min，N 越大越久）。runs `20261105-015025` … `20261105-063533-icpe-data-cw_main-*`。每個 block 7,344 列（6,174 個 job），已收進 `results/icpe_next/b/cw_main/<block>/`（commit 0aac6d9、5094c9c）。`cpu_residence_s_mean` 的 NaN 只出現在 LRU+NONE／O-ALL／O-DROP 沒有 CPU 駐留結束事件的列，和 H200 參照檔同樣的模式。
- **merge** `origin/icpe-next-sim` @ 03155d2 → `cc6b980`。程式＝7539f01（`git diff 7539f01 HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` 是空的；03155d2 只多 STATUS_SIM 與 G1 摘要）。
- **G0（merge 後）**：run `20261105-070841-icpe-data-g0-merged-7539f01`（32 worker），跑中。
- **CacheWise 的分析**：plan 跑完了，偏離 4 也已登記，所以照程序做。
  - run `20261105-071035-icpe-data-analyze-cw_main-ttl3`（`next_sim_analyze.py --pf-target cpu`，3 個 TTL），表在 `results/icpe_next/b/analysis/cw_main_ttl3/`（commit 2b33cdb）。
  - 先前的 run `…070937…` 作廢：沒有 export RUN_ID，run_id 欄寫成 `adhoc`。
  - **這張表是補跑前的**；上限 ≥10% 的格子，報告要用補跑後的值。
- **≥10% 補跑**（偏離 4 第 2 項）：run `20261105-071243-icpe-data-ttl7cells-cw_main` → `results/icpe_next/b/cw_main_ttl7/cells.csv`。
  - 選法：648 格 (trace, CPU, 慢層, GPU 閒／忙) 裡，有任何元件（含 O-ALL(report)）上限 ≥10% 的有 **225 格**。
  - 補跑內容：② 用其餘登記的 TTL **1／3／30／100 s**（TTLA、TTLB）× 10 個不預測的 CPU 規則（`x1_sim.NON_ORACLE`，程式會核對）。
  - 佇列：plan `cw_main_ttl7` 有 17 個 block（每個 (紀錄, 慢層組合) 一個），**19,840 個 job**。
  - sweep 不能只選 GPU 閒或忙，所以兩個都跑。多出來的列只會讓 ② 更強。
- 分析程式記下的 **violations 221 列**（④ 的元件比某個 ② 規則慢），照原樣記錄，沒有處理：
  - 慢的幅度中位數 0.5%、最大 6.4%。
  - 集中在 O-PF 對 LRU+RPF（92 列）、O-PF 對 LRU（57 列）、O-NOW 對 LRU+2HW（62 列）。

**接下來**（G0 PASS 之後，同一個佇列）：`next_b_queue.py run --plans cw_main_ttl7 tl_main --workers 32`。
- `cw_main_ttl7` 推算〔算術〕約 2–3 h：TTL 的 job 在 CacheWise 試跑每個 12–28 s。
- `tl_main` 推算約 32 h。
- 之後：`old_main` → `sa_main` → `bl_main` → `sens_rep`。每個 plan 跑完都做同樣的分析與 ≥10% 補跑。
- TraceLab 的補跑可能很貴：一個 job 約 200 s，每格 160 個 job。跑完 `tl_main` 先算格子數再排。

**需要 icpe-sim 做什麼**：沒有要求。violations 只是告知：O-PF 在部分格子比 LRU+RPF 慢，最多 6.4%（CacheWise）。

## 2026-11-05T01:52+08:00 — merge 09fd521、G0 再通過、正式模擬 `cw_main` 開跑

- **merge**：`git merge --no-ff 09fd521` → `a176e2d`（STATUS_SIM 2bad87e：G1 PASS on 09fd521，run `20261105-002207-icpe-sim-g1-09fd521`，4,416 列 × 29 欄 0 格不同、自檢 997／997）。只 merge 09fd521；7539f01（nb 記憶體修正＋prefix skip）的 G1 還在跑，過了再 merge。
- **G0（merge 後）PASS**：run `20261105-013010-icpe-data-g0-merged-09fd521`（程式 a176e2d、新功能全關、共用 venv、32 worker、1,193 s）：**4,416 列 × 29 欄＝128,064 格，0 個不同**。
- **正式模擬開跑**（條件：G1 PASS＋merge＋G0 PASS＋PREREG 偏離 4 已登記）：`python -I code/icpe/next_b_queue.py run --plans cw_main --workers 32`（setsid nohup，log `/ssd6/hungwei/icpe-next/runs/queue-cw_main-20261105-015025.log`）。第一個 block `20261105-015025-icpe-data-cw_main-cw_N16_s1`：6,174 個 job，開頭每個 6–15 s。12 個 block 依序：s1 的 N16→32→64→72，再 s2、s3。推算〔算術〕約 7 h（加速前的每 job 秒數）。
- 我**不看**正式輸出的上限數字，直到整個 plan 跑完、用 `next_sim_analyze.py` 照登記的判準算（偏離 4 已登記，所以這是照程序，不是保密）。

## 2026-11-05T00:42+08:00 — icpe-sim 09fd521（加速版）在 TraceLab 上的時間與記憶體（試跑，不進 results/）

| run | 程式 | 內容 | 每個 job（秒） | 記憶體 |
|:--|:--|:--|:--|:--|
| `20261105-002230-icpe-data-trial-tl-timing-09fd521` | 09fd521（git archive） | 和 `20261104-235459`（6b17191）同一批 10 個 job | LRU 156.0（舊 215.8）、O-BEL 156.7（舊 200.3）、×TTLA:10 197–229（舊 246–254）、O-PF 200–249（舊 237–288）、O-PF×TTLA:10 322–327（舊 301–304）＝**0.93–1.44 倍**；**40 個輸出欄逐值相同** | `/usr/bin/time` 最大 RSS **29.3 GB**（舊 6.8 GB） |
| `20261105-003309-icpe-data-trial-tl-mem-09fd521` | 09fd521 | TraceLab c1、CPU 64、LRU／O-BEL × TTLA:10／TTLB:10、8 worker；每 20 s 取 PSS | TTLB:10 126–136、TTLA:10 184–212 | 載入後 parent PSS 27.5 GiB；8 worker 時總 PSS 73.0 GiB、每個 worker 最多 8.3 GiB（`mem.log`） |

- 29 GB 的原因（icpe-sim 說明）：向量化的 x1_nb 每個鍵都變成新的 Python int；它正在修（共用 int 物件＋提早釋放），和另一個逐值相同的加速一起放進新 commit、另做 G1。
- **排法**（同使用者 00:15 的決定；icpe-sim 也這樣建議）：09fd521 的 G1 一過 → merge 09fd521 → G0 → 跑 `cw_main`（CacheWise 記憶體沒問題，0.44 GB／worker）。TraceLab 等修好記憶體的新 commit 過 G1 → merge → G0 → 跑 `tl_main`（照 09fd521 的量法，32 worker 約 210 GiB 以上，太緊）。
- 更新的推算〔算術〕：TraceLab 每 job 約 200 s（130–330）→ `tl_main` 3 × 6,174 × 200 ÷ 32 ≈ 32 h。

## 2026-11-05T00:25+08:00 — 方案 A 已登記（PREREG 偏離 4）；佇列程式；等 G1

- **PREREG 偏離 4**（`origin/icpe-next-a` 776e747；h200 用的是「偏離 4」，因為 3 已用在 W6）：內容＝下一節方案 A 的 1–4 點＋23:55／00:15 兩節的紀錄處理，已在 git 上核對。之後這裡提到方案 A 一律指偏離 4。
- 佇列程式 `code/icpe/next_b_queue.py`（commit 8579ec6 起）：plan＝`cw_main`（12）→ `tl_main`（c1→c2→c4）→ `old_main`（Mooncake 4、FI 6 週）→ `sa_main`（12）→ `bl_main`（3）→ `sens_rep`（代表格的 nvme807／只載入／wt／預取進 GPU／快取前綴／截斷 128K）。每個 block 一個 run 目錄（§4.1 的殼）、失敗就整個停、重跑會跳過做完的；`collect` 把彙總寫進 `results/icpe_next/b/<plan>/<block>/`（加 host 欄、xz）。
- 截斷 128K 的代表格紀錄（`--trunc-128k`：每個 session 遇到第一個 >131,072 token 的請求就截掉那一輪和之後）：CacheWise N16 s1 56,825 → 33,150 請求、SemiAnalysis N16 s1 98,827 → 40,799、FreeInference 第 1 週 → 6,313；TraceLab c1 轉換中。Mooncake（最長 126,195）、Bailian coder（最長 25,777）截斷後完全相同，不跑。
- **卡住什麼**：等 icpe-sim 的 G1（它說會在「目前 HEAD＋Engine 加速」的新 commit 上做，含 x1_nb；ac0c56e 那次已停掉、沒有比對）。G1 寫進 STATUS_SIM 後：merge → G0（32 worker）→ 開跑 `cw_main`。

## 2026-11-05T00:20+08:00 — 使用者：「11/9 要搞定」→ 縮格子方案 A（要 h200 登記偏離）

**使用者 11/5 原話**：「太久了 不行 目標5 天內搞定」「11/9 要搞定 注意喔」。之後在 AskUserQuestion 選了：
- **方案 A**：
  1. **CPU 規則 23 → 17**：拿掉 6 個事後穩健性版本 `O-DROP-H1800／-H3600`、`O-NOW-H1800／-H3600`、`O-ALL-H1800／-H3600`（PREREG §4 的 ④ 只列 O-PF、O-BEL／O-BELS、O-DROP、O-NOW、O-ALL）。17＝LRU、LRU2、ARC、S3FIFO、GDSF、PENS、FREQ2Q、LRU+NONE、LRU+2HW、LRU+2HA、LRU+RPF、O-BEL、O-BELS、O-DROP、O-NOW、O-PF、O-ALL。
  2. **TTL 值 7 → 3**：**0.3、10、300 s**（TTLA、TTLB 都是；0.3＝CacheScout、300＝SAGA 的 TTL_max、10＝對數中點）。GPU 規則＝LRU、BEL、TTLA:{0.3,10,300}、TTLB:{0.3,10,300}（8 個）。少了 TTL 值 → ② 只會變弱 → 上限只會偏高；**事後規則（現在登記）：主結果裡 ④ 對 ② 上限 ≥10% 的每一格，用原登記的 7 個 TTL 補跑 ②，報告用補跑後的值**。
  3. **所有登記的紀錄變體都跑主結果的主時間設定**（cake × ssd／b32／nfs × GPU 閒／忙）：CacheWise 12（前綴＝結構推得）、TraceLab 3（c1／c2／c4，前綴＝結構推得）、SemiAnalysis 12、Bailian 3（traceA、thinking、coder）、Mooncake 4、FreeInference 6 週；CPU 9 個容量照 PREREG §4。每個變體 6,174 個 job。
  4. **敏感度只跑代表格**（不是全部格子）：每份紀錄 1 個變體——CacheWise N16 s1、TraceLab c1、SemiAnalysis N16 s1、Bailian coder、Mooncake conversation R1 與 toolagent R1、FreeInference 第 1 週（2026-05-17_2026-05-23）——× CPU **111.0625、128 GiB**，跑：nvme807（cake）、只載入（4 個慢層）、`wt`、預取進 GPU、快取命中前綴（只 CacheWise／TraceLab）、截斷 128K（PREREG §4 敏感度 1）。
- **CPU 維持 32 worker**。
- 預估〔算術，32 worker、沒加速〕：主結果約 84.8 h（CW 7.1、TL 40.2、SA 31.2、其他 6.4）＋代表格約 15 h ≈ **99 h**；icpe-sim 的 Engine 加速出來就會縮短。W4、W3、第二批功能另算。

**需要 h200 做什麼**：把上面 1–4 寫進 PREREG「偏離 3」（使用者 11/5 在 icpe-data 決定；我這邊**還沒看任何正式結果**——正式模擬還沒開始，G1 還沒過）。也請一併登記前一節的：CacheWise 5 份＝400 個對話、前綴主結果＝結構推得／敏感度＝快取命中、resume 複本不去重、Bailian traceB 跳過。我會等偏離 3 出現在 `origin/icpe-next-a` 才分析任何正式輸出。

**需要 icpe-sim 做什麼**：同上一節（Engine 加速）；另外 W4（7fd39f3）過 G1 後告訴我。

## 2026-11-05T00:15+08:00 — 使用者決定排程；請 icpe-sim 加速 Engine

**使用者 11/5 的決定**（AskUserQuestion，這個 session）
1. 排程：**G1 一過就先跑 CacheWise 主結果·主時間設定**（12 個變體，約 21 h）；**同時請 icpe-sim 加速 Engine**（結果要逐值相同、用 G1 驗）；CacheWise 跑完接 **TraceLab c1 → c2 → c4**；加速版通過驗證就換用。其餘照順序接在後面（給 CCGrid）。
2. Bailian traceB：**跳過**，論文寫明「traceB 沒有多輪結構（172,800 個請求全部 turn＝1、parent＝−1）」。Bailian 只跑 traceA、thinking、coder。

**需要 icpe-sim 做什麼**（使用者決定的）：**加速 `x1_sim` 的 Engine**，條件是新功能關閉時 G1 逐值相同、開啟時和 ac0c56e（或 G1 通過的那個 commit）逐值相同。量測基準：TraceLab c1、CPU 64、GPU 111.0625 wb，LRU／O-BEL／O-PF × GPU LRU／TTLA:10 每個 job 200–304 s、每個 worker 4.75 GB（run `20261104-235459-icpe-data-trial-tl-timing`，可直接用 `/ssd6/hungwei/icpe-next/trial/x1_nb/tracelab_c1.npz`）。通過後在 STATUS_SIM 寫 commit 與驗證 run_id，我 merge 後再做一次 G0 再換。

**正式跑的輸出**：原始輸出（sweep CSV、逐請求 TTFT 的 npz——`x1_boot.py` 的成對 bootstrap 要用）在 `/ssd6/hungwei/icpe-next/runs/<run_id>/`；git 的 `results/icpe_next/b/` 只放彙總（sweep CSV 加 `host=acvlab` 欄、xz 壓縮，和分析表）。npz 每個 job：CacheWise 約 0.4 MB、TraceLab 約 4.4 MB（TraceLab 主時間設定 3 個變體約 240 GB；/ssd6 剩 2.7 TB）。

## 2026-11-05T00:10+08:00 — 試跑計時完成；全部正式格子的 job 數與預估總時間

**試跑計時**（全部用 icpe-next-sim 6b17191 的 git archive、GPU 層 111.0625 GiB wb；輸出不進 results/；機器 load 65–87／96）
| run | 紀錄 | 內容 | 每個 job（秒） | 每個 worker RSS |
|:--|:--|:--|:--|:--|
| `20261104-235049-icpe-data-trial-cw-timing` | CacheWise N16 s1（943 萬 chunk） | CPU 64；LRU、GDSF、O-BEL、O-DROP、O-PF、O-ALL、LRU+RPF × GPU LRU／BEL／TTLA:10／TTLB:10；16 視角 LRU、O-BEL | 7.6–13.7（TTLA:10 約 12.5；16 視角 8.0–9.1） | 0.44 GB |
| `20261104-235858-icpe-data-trial-cw-timing2` | 同上 | CPU 8、1024；LRU2、PENS、ARC、S3FIFO、O-NOW、O-ALL-H3600 × GPU LRU／TTLA:0.3 | GPU LRU 7.6–9.5；**TTLA:0.3 15.8–28.1** | 0.44 GB |
| `20261104-235922-icpe-data-trial-cw-timing3` | 同上 | CPU 1024；O-BEL、O-BELS × GPU LRU／BEL | 7.1–9.0 | 0.44 GB |
| `20261104-235459-icpe-data-trial-tl-timing` | TraceLab c1（2.23 億 chunk、665,453 請求） | CPU 64；LRU、O-BEL、O-PF × GPU LRU／TTLA:10 | **200–304**（中位約 250）；載入約 3.5 分鐘（舊版純 Python x1_nb；icpe-sim 3140f3c 已向量化） | **4.75 GB**（parent 峰值 6.8 GB） |

每個 chunk 存取的成本兩份紀錄幾乎相同：CacheWise 11 s ÷ 943 萬＝1.17 µs，TraceLab 250 s ÷ 2.23 億＝1.12 µs〔算術〕→ SemiAnalysis（4,221 萬 chunk）用 1.15 µs/chunk 推算＝48.5 s/job（**推算，沒量**）。

**job 數**（6b17191 的 `jobs_for` 數的；每個紀錄變體）：主結果（wb、預取到 CPU）全部時間設定＝49,014（分三批跑，結果和一次跑相同：① 主時間設定 cake×{ssd,b32,nfs}×{閒,忙}＝**18,342**、② nvme807×cake＝6,318、③ load×4 慢層＝24,354）；`wt` 再 49,014；預取進 GPU 11,520（其中主時間設定 4,320）。

**預估總時間**〔算術：job 數 × 每 job 秒 ÷ 32 worker；每 job 秒的來源見最後一欄〕
| 紀錄 | 變體 | 每 job 秒 | 主·主時間設定 | 主·全部時間設定 | wt | 預取進 GPU | 前綴＝快取（CW／TL） | 每 job 秒的來源 |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| CacheWise | 12 | 11.0 | 21.0 h | 56.2 h | 56.2 h | 13.2 h | 56.2 h | 試跑實測 7.6–28.1 s，加權估 11 |
| TraceLab | 3 | 250.0 | 119.4 h | 319.1 h | 319.1 h | 75.0 h | 319.1 h | 試跑實測 200–304 s（CPU 64；TTLA:0.3 沒量） |
| SemiAnalysis | 12 | 48.5 | 92.7 h | 247.8 h | 247.8 h | 58.3 h | — | 推算：1.15 µs/chunk |
| Bailian（traceA、thinking、coder） | 3 | 2.0 | 1.0 h | 2.6 h | 2.6 h | 0.6 h | — | 推算（沒量） |
| Mooncake | 4 | 20.7 | 13.2 h | 35.2 h | 35.2 h | 8.3 h | — | G0 舊程式實測（沒開 GPU 層） |
| FreeInference（6 週） | 6 | 5.0 | 4.8 h | 12.8 h | 12.8 h | 3.0 h | — | H200 T1／T5 每 job 中位 2.3–4.1 s（別台） |
| **合計** | | | **252 h** | **674 h** | 674 h | 158 h | 375 h | |

- 第一批全部（主＋wt＋預取進 GPU＋快取前綴）≈ **1,881 h ≈ 78 天**（32 worker）。還沒算：截斷 128K 的敏感度（每份紀錄再一份主結果）、第二批（慢層容量、block 粒度 APC 重算、排隊、W4 雜訊）＝NOT_ESTIMATED（程式還沒有）。
- ICPE 要的：CacheWise 主·主時間設定 **21 h**；TraceLab 主·主時間設定 **119 h**（只跑 c1＝原始到達 **40 h**）。從 11/5 00:10 到 11/9 20:00 約 116 h。
- 記憶體：TraceLab 32 個 worker × 4.75 GB ≈ 152 GB（這台可用約 324 GB、swap 已滿）。

**需要 icpe-sim 做什麼**：你提過可以加速 Engine（結果要逐值相同）——TraceLab 的單一 job 是 4 分鐘級、全部格子是 78 天級，這是關鍵路徑。要不要做、先做哪個由使用者決定（我會問）。

**需要使用者決定什麼**：ICPE 前跑不完的部分怎麼取捨（見下一個問題）；Bailian traceB。

## 2026-11-05T00:00+08:00 — G0 通過；Bailian 兩個新發現

**做完什麼**
- **G0 PASS**：run `20261104-232902-icpe-data-g0-main`（程式＝`origin/main` 77c25f6 的 `x1_sim.py`／`p3_fvar.py`，code_sha `c5c5b1cd1b67`；共用 venv；`X1_TRACE_DIR=/ssd8/hungwei/paper-hkv/datasets/mooncake/traces`；另需 `M7_RES=<repo>/results/m7_write_policy_h200`；16 worker、1,722 s、exit 0）。和 `results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz` 比：**4,416 列 × 29 欄＝128,064 格，0 個不同**（`code/icpe/next_b_g0.py`：去掉 run_id、ts、elapsed_s、code_sha，依 7 個鍵排序逐字串比；輸出 `…/g0_compare.txt`）。→ 環境（Python 3.12.15／numpy 2.3.5／pandas 3.0.5／pyarrow 25.0.1）與 main 的程式重現 H200 T1 V1，不用再拿 6ff7584 重跑（main 的 x1_sim.py 比 6ff7584 多的 +84 行在預設下不影響輸出）。
- Bailian（commit `b916bbd`）：
  - **hash_ids 是逐塊 hash，不是鏈式**（上游 README：每個 16-token 塊各自做 salted SipHash-2-4）。traceA 的鏈式檢查違反 301,710 次、traceB 893,399 次（coder、thinking 0 次）。→ chunk 鍵改成「從頭到這一塊的整段 hash 序列」的編號（和 SemiAnalysis 相同）；鏈式違反數只記錄。改完 traceA 的規則 6 全過（parent 前綴 1.0000、間隔÷輸出 0.29 s/token）。
  - **traceB 沒有任何多輪結構**：172,800 個請求全部 turn＝1、parent_chat_id＝−1（type 87% 是 `api`）→ 依資料定義沒有「回來的請求」，主指標無定義；載入函式的 parent 前綴檢查 0/0 → 停（run `20261104-235631-icpe-data-conv-bailian`，錯誤：`[bailian-traceB 前導共用塊 ≥ parent 的完整塊] 規則 6 的驗算比例 nan < 0.9 → 停`）。traceA、thinking、coder 已轉好。
- 轉好的紀錄（`/ssd6/hungwei/icpe-next/datasets/x1_nb/`，不進 git）：CacheWise 24 份（N 16／32／64／72 × seed 1／2／3 × 前綴 struct／cache；每份 56,825 請求、540 session、943 萬 chunk；N16 跨 364–416 h、N64／72 約 200–215 h）、Bailian traceA／thinking／coder、TraceLab 與 SemiAnalysis 轉換中。

**需要使用者決定什麼**：Bailian traceB（沒有多輪）要跳過，還是像 Mooncake 那樣用 hash 前綴重建 session？（我會和總時間一起問。）

## 2026-11-04T23:55+08:00 — 資料清點、下載、載入函式與轉換、資料陷阱、使用者決定、試跑計時（G0 還在跑）

**做完什麼**

1. **資料清點**（這台既有的都在 `/ssd8/hungwei/paper-hkv/datasets/`，`/ssd6/hungwei/icpe-next/datasets/` 用 symlink 指過去；/ssd8 剩 299G、96% 滿，所以新下載的放 /ssd6）

| 紀錄 | 位置 | 版本 | 完整性（今天重算） |
|:--|:--|:--|:--|
| Mooncake | /ssd8（symlink） | commit d750cdda | conversation `b8cbb061…`、toolagent `48a2db1a…`＝README.txt＝H200 runner 檢查的值 |
| CacheWise | /ssd8（symlink） | HEAD 181c435a（`git rev-parse` 確認；LFS 檔 0 個，資料直接在 git 裡） | git 物件雜湊；工作樹乾淨 |
| Qwen-Bailian | /ssd8（symlink） | 5f7439c5 | 4 個 jsonl 的 sha256＝README.txt（＝下載時的 LFS oid） |
| FreeInference | /ssd8（symlink；**沒重下載**） | 16d3c04d | 48 個 parquet 的 sha256 全部＝下載 log 裡對過 HF LFS oid 的值（`sha256sum -c`，0 個不同） |
| TraceLab | **新下載** /ssd6 | release v0.0.2 | sha256 `11ce51ec…`＝GitHub release digest（run `20261104-233022-icpe-data-dl-tracelab-semianalysis`，用 `setup_3090_data.sh do_tracelab`） |
| SemiAnalysis | **新下載** /ssd6 | HF 23f152f6 | `hf_verify … files=3 bad=0`（同一個 run，`do_semianalysis`） |

2. **載入函式與轉換**（commit `3641092`）：`code/icpe/agent_traces.py` 加 `load_cachewise()`、`load_bailian()`（規則 6 斷言寫在函式裡），`load_tracelab()` 補 `out／provider／trigger` 欄（原欄位不變）；`code/icpe/next_b_convert.py` 轉成 icpe-sim 的 `nb` 格式（`code/p3x/x1_nb.py`，icpe-sim 6b17191；我照用，不要求改）；`code/icpe/next_b_g0.py` 對拍程式。轉好的 npz 在 `/ssd6/hungwei/icpe-next/datasets/x1_nb/`（**不進 git**）。
   - 規則 6 驗算（CacheWise，全部過）：同一 requestId 的輸入三欄相同 0.9991；output_tokens 不減 1.0000；cache_read(k) ≤ inp(k−1)+out(k−1) 0.9927；間隔 <10 s 時 cache_read(k) ≥ 0.9×inp(k−1) 0.9857；(end−start)/out 中位數 31.7 ms/token。Bailian coder：len(hash_ids)＝ceil(input/16) 1.0000；鏈式前綴違反 0；前導共用塊 ≥ parent 完整塊 0.9931；間隔÷上一輪輸出 0.199 s/token。TraceLab：prefix+new＝input 1.0000。

3. **資料陷阱（§5.7 以外新發現的）**
   - **供應商快取會過期**：TraceLab 的 `prefix_tokens`（Claude 列＝`cache_read`，1.0000 相同）和 CacheWise 的 `cache_read` 都是「供應商快取命中」，5 分鐘／1 小時沒用就掉成 0。間隔 ≥1 h 時 `prefix ÷ 上一輪輸入` 中位數：TraceLab Claude 0.022、Codex 0.059、CacheWise 0.053；但 `這一輪輸入 ÷ (上一輪輸入＋輸出)` 中位數都 ≈1.00（上下文是接續的）。間隔 <60 s 兩者一致（中位數 1.000）。→ 照 NEXT_PLAN §5.6 字面用快取命中，長間隔回來的請求會被當成「沒有可還原的前綴」。
   - **resume 複本**：Claude Code resume 會把舊對話整段複製進新檔。CacheWise：25 對 session 的前段請求序列（inp, cache_read）完全相同、長 18–356 個請求，**1,885／11,365 個請求（16.6%）是複本**；複本那段的時間戳有一部分被改成 resume 當下的時間。TraceLab：**14,685／665,453（2.2%）**屬於 5,833 組時間到毫秒都相同的複本（全是 Claude，129 個 session）。上游作者的處理：CacheWise `extract_events.py` 不去重；TraceLab 只用 `(provider, session_id, round_id)` 去重（換了 session_id 的複本不會被去掉）。
   - **CacheWise 可用對話＝80**（不是 94 減 1）：12 個原始檔是 0 byte，2 個沒有任何 LLM 呼叫（session_0061 只有 system 節點）。108 個 session＝80 主＋28 子 agent。
   - CacheWise 同一個 requestId 會記成好幾個 assistant 節點（串流，每個 content block 一個）；最後一個的 output_tokens 有時仍是佔位值 1（模擬器不用輸出長度，不影響）。順序要用 requestId 序號：用 parent 時間排會排錯 128 個（resume 改寫了時間）。
   - 一個請求的送出時間＝第一個 assistant 節點的 parentUuid 指向的 user／tool_result 節點時間（`llm_call.timestamp` 是回應寫下的時間，不是送出時間）。

4. **使用者 11/4 的決定**（AskUserQuestion，這個 session）
   - 前綴規則：**兩種都跑**——主結果＝結構推得（輸入沒變短、中間沒有 compact_boundary → 上一輪整段輸入都算前綴；否則退回快取命中），敏感度＝快取命中（NEXT_PLAN §5.6 原文）。
   - resume 複本：「不確定的都看類似 paper 怎麼做」→ 上游作者都不做跨 session 去重 → **照釋出的資料用、不去重**，在論文限制寫明 16.6%／2.2% 是複本。
   - CacheWise 複本數：「看其他人怎麼做」→ Continuum 把紀錄重複 1–5 次、PREREG 登記 5 份 → **5 份＝400 個對話**（PREREG §3 D 的「470」是用 94 算的）。
   - CPU：**固定 32 個 worker**（`nice -n 10`）。

5. **試跑計時**（run `20261104-235049-icpe-data-trial-cw-timing`；程式＝icpe-next-sim 6b17191 的 git archive，**沒 merge、不進 results/**）：`nb:cachewise_N16_s1`（56,825 請求、540 session、943 萬 chunk）、CPU 64 GiB、GPU 層 111.0625 GiB wb、16 worker、當時 load ≈ 87／96。每個 job **7.6–13.7 s**（LRU／GDSF／O-BEL／O-DROP／O-PF／O-ALL／LRU+RPF × GPU 的 LRU／BEL／TTLA:10／TTLB:10；TTLA 最慢約 12–13.6 s；16 個時間設定的多視角 job 8.0–9.1 s）；載入約 10 s；每個 process RSS 約 440 MB。**沒涵蓋**：最重的規則（LRU2、PENS、ARC、S3FIFO、O-ALL-H）、其他 CPU 容量。

6. **正式格子的 job 數**（照 PREREG，直接呼叫 6b17191 的 `jobs_for` 數，不是估）：每個紀錄變體（一個 trace × 到達設定）
   - 主結果（wb、預取到 CPU）：**48,258 個 job**＝9 個 CPU 容量（8、16、32、64、128、256、512、1024、111.0625）× 23 個 CPU 規則 × 16 個 GPU 規則（LRU、BEL、TTLA×7、TTLB×7）× 時間設定（慢層 ssd／b32／nfs／nvme807 × GPU 閒／忙 × cake／load）；其中 TTL 類 36,288、預取類 11,520（都是一個時間設定一個 job）。
   - `wt` 敏感度：再 48,258；預取進 GPU 的敏感度：11,520。
   - 紀錄變體數：CacheWise 12（N 16／32／64／72 × seed 1／2／3）× 前綴規則 2；TraceLab 3（壓縮 1／2／4×）× 前綴規則 2；SemiAnalysis 12；Bailian 4；Mooncake 4（conversation／toolagent × R1／R2）；FreeInference 6 週（T1／T4／T5 用的那 6 週；CPU 用 PREREG §4 的清單＝舊網格的聯集＋111.0625）。
   - **推算〔算術，用上面一格的試跑時間，不是量的〕**：CacheWise 主結果（結構前綴、wb、預取到 CPU）12 × 48,258 ≈ 57.9 萬個 job × 約 10 s ≈ 579 萬 CPU-秒 → 32 worker 約 **50 小時**。TraceLab 的試跑計時還在做（轉換中；它有 665,453 請求、2.23 億 chunk，約 CacheWise 的 24 倍），結果出來再補全部總時間。

**卡住什麼**：G0 還在跑（run `20261104-232902-icpe-data-g0-main`，用 main 的程式）。正式模擬要等 icpe-sim 的 G1（STATUS_SIM）＋我 merge 後再做一次 G0。

**需要 icpe-sim 做什麼**：無（nb 格式照用）。

**需要 h200 做什麼**（使用者已在這個 session 決定，請登記 PREREG 偏離）：(1) CacheWise 5 份複本＝**400** 個對話（不是 470；可用對話 80）；(2) TraceLab／CacheWise 的前綴：主結果＝結構推得，快取命中當敏感度；(3) resume 複本不去重（照上游作者）。

**需要使用者決定什麼**：全部格子的總時間出來後（TraceLab 計時），ICPE 前跑不完的部分怎麼取捨，會再問。

## 2026-11-04 23:3x（Asia/Taipei）— venv ready

**做完什麼**
- worktree：`/ssd6/hungwei/icpe-next/data`，分支 `icpe-next-data`（從 `origin/main` 77c25f6 開），已 push。來源 clone＝`/home/hungwei/llm/POC/paper-hierarchical-kv-state`（和 icpe-sim 的 `sim/` 同一個 repo）。
- **venv ready**：`/ssd6/hungwei/icpe-next/venv`（run `20261104-232611-icpe-data-venv`，目錄 `/ssd6/hungwei/icpe-next/runs/20261104-232611-icpe-data-venv/`，exit 0）
  - Python **3.12.15**（`3.12.15 (main, Oct 21 2026, 17:47:06) [Clang 23.1.3 ]`）
  - numpy **2.3.5**、pandas **3.0.5**、pyarrow **25.0.1**；相依：python-dateutil 4.0.0、tzdata 2026.6
  - 用法：`/ssd6/hungwei/icpe-next/venv/bin/python -I ...`（或 `source /ssd6/hungwei/icpe-next/venv/bin/activate`）
- 建法（為什麼和平常不一樣）：
  - 這台的 uv 0.12.3 下載清單最高只到 3.12.13 → 用 `uvx --from uv==0.13.7 uv ...` 臨時跑較新的 uv；`~/.local/bin/uv` 沒動。
  - 受控 Python 裝在**獨立目錄** `/ssd6/hungwei/icpe-next/uv-python/`，沒有裝進共用的 `/ssd8/hungwei/paper-hkv/uv-python/`：那裡的 `cpython-3.12-linux-x86_64-gnu` 是 minor 版 symlink，`venv/tiara` 的 `home` 就指著它；裝 3.12.15 進去會讓既有 venv 被透明升級。`venv/tiara`、`venv/vllm028` 沒動。
  - `uv python install --no-bin`：沒有改 `~/.local/bin/python3.12`（它指向別的專案）。

**還沒驗證**
- H200 那次 run（`20261028-071007-icpe-t1-x1-sweep-V1`）的 context 不在 git 裡，我看不到它的 python-dateutil／tzdata 版本：NOT_VERIFIED。核心四項（Python、numpy、pandas、pyarrow）照 NEXT_PLAN §3b 的版本。若 G0 對不上，這是要先查的一項。

**卡住什麼**：無。
**需要 icpe-sim 做什麼**：可以開始用 venv 做 G1。
**需要使用者決定什麼**：無（目前）。
