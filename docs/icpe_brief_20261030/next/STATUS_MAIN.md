# STATUS_MAIN（icpe-main，acvlab，分支 `icpe-next-main`，worktree `/ssd6/hungwei/icpe-next/main-ana`）

> 只由 `icpe-main` 寫。格式照 NEXT_PLAN §3b：日期時間／做完什麼（commit、run_id）／卡住什麼（完整錯誤）／需要別的 session 做什麼／需要使用者決定什麼。最新的一節在最上面。
> icpe-main＝**正式的主結果分析 session**：算正式的上限表、照 PREREG 下判定。不跑模擬、不動任何佇列、不碰 `/ssd6/hungwei/icpe-next/data`、不改別人的檔（`next_sim_analyze.py` 原樣使用）。自己的輔助腳本＝`code/icpe/main_*.py`。

## 2026-11-06T09:05+08:00 — 論文的兩個 Mooncake 舊測試：照原定義在新、舊模擬器上重算並列（使用者決定）

**使用者決定**（AskUserQuestion，11/6 09:0x，本 session）：論文的 50.5%（conversation 預取、CPU 128 GiB）與「192 格裡截尾 H3600＋排除寫入飽和後 36 格 ≥15%」（EVIDENCE B9）——**新模擬器照原定義重算，和舊值並列**；**完全照舊定義**（rule=min、4 個元件、CPU 128–1,024 GiB、192 格），另外並列 R1／R2 與 O-GPU（只報告，不改偏離 7 的判定）。

**做完什麼**：`code/icpe/main_oldtests.py`；run `20261106-090440-icpe-main-oldtests-verA`（第一次 `…-090416-…` 因我的腳本查表錯誤 exit 1，留了 `FAILED_SCRIPT_BUG`，修正後重跑）→ `results/icpe_next/main/oldtests_verA/`。
- **定義核對 PASS**：在舊 V0 表（論文數字的來源 `results/icpe_h200/t1/p01/sim_V0/x1/`）上重現 50.5%、95／192（依 CPU 28／31／24／12、依慢層 nfs 42／ssd 29／b32 24）、H3600＋不飽和 36（16／12／8／0；ssd 19／b32 17／nfs 0）。
- **發現**：論文的數字來自 **V0** 常數（舊模擬器 sim_V0），新模擬器用 PREREG §4 的 **V1**。所以並列三欄：old_V0（論文）、old_V1（同一舊模擬器、V1 常數）、new（GPU 層，版本 A、3 個 TTL，不是正式值）。

| 測試（rule=min、Cake） | old_V0（論文） | old_V1 | new 版本 A |
|:--|--:|--:|--:|
| T1：conversation O-PF、SSD、CPU 128 GiB | 50.5% | 53.2% | 39.1% |
| T2：192 格 ≥15%（不截尾） | 95 | 97 | 113 |
| T2：不截尾、只算不飽和 | 62 | 65 | 79 |
| T2：截尾 H3600＋只算不飽和 | 36 | 38 | NOT_MEASURED（要 old_p1 的截尾列） |

- 新模擬器不截尾的 113 格裡 O-DROP 41 格；舊模擬器截尾 H3600 後 O-DROP 是 0 格（B2：大半是紀錄太短）。所以新的截尾後格數要等版本 B，現在不能推。
- 只報告：新模擬器的 O-GPU 在同一組 48 格（2 紀錄 × 4 容量 × 3 慢層 × 2）裡 36 格 ≥15%（rule=min）。

## 2026-11-06T08:58+08:00 — 更正：本檔與 PREREG 偏離 7 的時間標記（icpe-data 指出）

- **錯在我**：下面各節原本的節名時間（09:00、09:05、09:15）和「使用者決定 09:0x」是我估的，沒有用 `date`。icpe-data 在 SendMessage 指出：我傳訊息之後 acvlab 的鐘才 08:56。我傳給 icpe-data 的訊息也寫了「09:05 節」，那就是本檔現在的 08:54 節。
- 用 git 與檔案時間核對後的實際時間：開跑版本 A 08:48:46；版本 A 跑完 08:50:41（`exit_code` 的 mtime）；使用者選「commit＋push」與「偏離 7 由 icpe-main 寫」在 08:51–08:53（回報版本 A 之後、偏離 7 的 `date -u` 00:53:20Z 之前）；commit 9cba644／bcd0bad 08:54:39；commit 91a4d9d 08:56:14。下面各節的節名已改成實際時間。
- **先後順序不變**：使用者是在看過版本 A 之後才決定登記；做決定 1–3（08:4x，08:46:20 merge 之前）時只看過 manifest 的摘要。
- 偏離 7 的同步（git 核對）：`origin/icpe-next-data` @ b46635e、`origin/icpe-next-a` @ 1c0973f 的 `PREREG.md` 都和 bcd0bad 逐字相同。
- PREREG 偏離 7 原文寫的「09:0x」也是錯的。PREREG 只往後加，所以在檔尾加了一節「偏離 7 更正」（`date -u` 2026-11-06T00:57:25Z），不改原文。要不要 push、其他分支要不要再同步：等使用者決定。

## 2026-11-06T08:56+08:00 — 偏離 7 的同步狀態

- h200 回覆（SendMessage）：已核過、已同步。我在 git 上確認：`origin/icpe-next-a` @ 1c0973f 的 `PREREG.md` 和 `icpe-next-main` @ bcd0bad **完全相同**（`git diff` 是空的）。h200 下一個偏離編號是 8。
- `origin/icpe-next-data` 的 `PREREG.md` 還**沒有**偏離 7。已用 SendMessage 提醒 icpe-data；要不要同步由 icpe-data 決定。之後合併分支時要注意兩邊的 PREREG 一致。
- P1：背景每 15 分鐘 fetch 一次，等全部 P1 block 進 git（cw 24、tl 6、old 4、bl 2、sa 24）。

## 2026-11-06T08:54+08:00 — PREREG 偏離 7（判定方式的釐清）已寫；版本 A 進 git

**使用者決定**（AskUserQuestion，11/6 08:51–08:53：版本 A 08:50:41 跑完、回報之後，偏離 7 的 `date -u` 00:53:20Z 之前）
- 版本 A：全部 commit，push `icpe-next-main`。
- 判定規則（原始／合成到達各下判定字、Mooncake R1／R2 各自判定、寫入飽和照算另標）：**登記成 PREREG 偏離 7，由 icpe-main 寫**。

**做完什麼**
- **PREREG 偏離 7**（acvlab `date -u` 2026-11-06T00:53:20Z）：只在檔尾加一節（25 行，0 行刪除）。內容＝上面三點、icpe-main 照既有文件的解讀（算判定的元件、工具類的分法、「≥15%」照字面、簡單線索 NOT_MEASURED → 推翻條件「未定」），以及寫的當下已經看過的正式結果。
- **版本 A 的表**：`results/icpe_next/main/verA/`（逐格上限表與 violations 用 xz，解壓後和 run 目錄的原檔 `cmp` 相同；`manifest.json` 有 cmd、context、未壓縮檔的 sha256）。

**需要 h200 做什麼**：偏離 7 在 `icpe-next-main` 的這個 commit。要不要同步進 `icpe-next-a` 由你決定（你同步偏離 5、6 的做法：只取 `PREREG.md` 一個檔）。**下一個偏離編號是 8**。

**需要 icpe-data／icpe-trx 做什麼**：P1 的 block 照常 collect、commit、push 就好；我 fetch 後照 PREREG 偏離 6 算版本 B（主結果＋`_p1`，兩邊同一組 7 個 TTL；偏離 5 的算法並列；`--dead-h 1800`／`3600` 只報告），先核列數。`cw_main_ttl7` 不放（和 P1 重疊）。

**接下來**：等 `{cw,tl,old,bl}_p1` 與 `sa_p1` 全部進 git。

## 2026-11-06T08:51+08:00 — 版本 A（3 個 TTL）完成：對拍 6／6 相同、格數自檢 PASS；判定（版本 A，不是正式值）

**run** `20261106-084846-icpe-main-verA-ttl3`（`/ssd6/hungwei/icpe-next/runs/` 下；cmd.sh、context.txt、stdout／stderr、exit_code＝0；git HEAD bf7915f；牆鐘 1:40、最大 RSS 338 MB；nice 19、單一行程）。
- `next_sim_analyze.py` 原樣，`--pf-target cpu`，6 類紀錄分開跑（避免 `rule=min` 複製單一規則的紀錄）；CacheWise 另跑偏離 5 版（`cw_main`＋`cw_main_ttl7`＋`--rival-only-gpu` 8 個 TTL，內建自檢 PASS）。p90 另跑，只報告。
- **對拍**（`out/cmp.csv`）：重算的 cw 3 TTL（headroom 3,888、violations 221）、cw 偏離 5（3,888、369）、sa 3 TTL（3,888、5,322）和 git 上的 `analysis/cw_main_ttl3`、`cw_main_final`、`sa_main_ttl3` **逐格相同**（去掉 run_id、ts、host）。
- **格數自檢**（`main_summary.py`）：42 個 (trace, rule) × 每個元件都是 54 格，PASS。
- **版本標記**：`A_report`＝偏離 6 的解讀（CacheWise 用偏離 5 版；TraceLab、SemiAnalysis、Mooncake、FreeInference、Bailian 是 3 個 TTL、**沒做過 ≥10% 補跑**——補跑被 P1 取代——所以 ② 偏弱、上限只可能偏高）；`A_ttl3`＝全部 3 個 TTL。② 不含 block 粒度 APC（偏離 2 第 3 點）。**正式值等 P1（版本 B）。**

**判定（版本 A_report；`out/verA_report_verdict.csv`）**〔模擬〕
| 到達 | 類別 | 變體數 | 有格子 ≥15% 的變體 | ≥15% 這一半 | 推翻條件 |
|:--|:--|--:|:--|:--|:--|
| 原始 | 工具類 | 9 | 3：Mooncake toolagent R1、R2；TraceLab c1（FreeInference 6 週最大 0.9%） | 成立 | 未定（簡單線索 W3 NOT_MEASURED） |
| 合成 | 工具類 | 26 | 21：SemiAnalysis 12／12、CacheWise 7／12（N32 s2、N64、N72）、TraceLab c2、c4 | 成立 | 未定（同上） |
| 原始 | 非工具類（只報告） | 5 | 5：Bailian 3、Mooncake conversation R1、R2 | — | 不適用 |

- 寫入飽和：≥15% 的格子裡只有 Mooncake 有（conversation 69、toolagent 72 格），其他紀錄 0。
- CacheWise 3 TTL 與偏離 5 版：算判定的 5 個元件 ≥15% 格數相同（O-GPU 111、O-ORD 48、O-PF 31）。

## 2026-11-06T08:49+08:00 — 開工；使用者對 4 個分析選擇的決定；版本 A（3 個 TTL）開算

**讀了什麼**：CLAUDE.md、NEXT_PLAN_20261104、PREREG（§1、§3、偏離 1–6、偏離 6 補記）、STATUS_DATA（到 08:25）、STATUS_SIM（到 06:14）、STATUS_A（到 07:55）、STATUS_ANA（`origin/icpe-next-ana` 279eb7e，到 03:05）、STATUS_TRX（`origin/icpe-next-trx` 87a5a86，到 08:12）、`code/icpe/next_sim_analyze.py`。

**merge**：`origin/icpe-next-trx` @ 87a5a86 → a586e83（只加 `sa_main` 12 個 block、`analysis/sa_main_ttl3`、`sa_main_ttl7/cells.csv`、STATUS_TRX；沒有程式改動）。

**使用者的決定**（AskUserQuestion，11/6 08:4x，本 session；都是 PREREG 沒寫到的分析選擇）
1. 版本 A 照提的方式開始算。
2. **判定範圍**：推翻條件的「真實的工具類紀錄」——**原始到達與合成到達各下一個判定字**，不合併。
   - 原始到達：TraceLab c1、Mooncake、FreeInference、Bailian。合成到達：CacheWise、SemiAnalysis（固定並行 N）、TraceLab c2／c4（開始時間壓縮）。
   - **提問時我已經看過 SemiAnalysis N16 s1（合成到達）有 ≥15% 的格子**（見下面「已看過」），問題裡有寫明。
3. **Mooncake**：**R1、R2 各自下判定**（不是以前慣例的 `rule=min`；`rule=min` 的列照程式輸出、只報告）。
4. **寫入頻寬飽和的格子**（`write_saturated`）：照算、另外標格數，不排除。
- 工具類的分法（不是新決定，照 NEXT_PLAN W2／§5.2）：TraceLab、CacheWise、SemiAnalysis、FreeInference、Mooncake toolagent＝工具類；Mooncake conversation、Bailian＝只報告。
- 要不要把 2、3 登記成 PREREG 的釐清：→ 已登記為偏離 7（見 08:54 那節）。

**已看過哪些正式結果**（照偏離 5 的慣例照實列出）
- 開工前讀 manifest 時，`stdout_tail` 欄裡的摘要表（我只印了前 600 字元）：
  - `results/icpe_next/b/analysis/sa_main_ttl3/manifest.json`（run `20261106-074822-icpe-trx-analyze-sa_main-ttl3`）：SemiAnalysis N16 s1 的 O-ALL(report)、O-DROP、O-GPU、O-NOW、O-ORD 五列（格數、最大、中位數、≥15%／≥5% 格數）。O-GPU、O-ORD、O-ALL(report) 有 ≥15% 的格子；O-DROP、O-NOW 最大 <3%。
  - `analysis/cw_main_ttl3`、`cw_main_final` 的 manifest：CacheWise N16 s1、s2 的部分元件列，最大都 <1.2%、0 格 ≥5%。
- 讀 sweep 時只看了欄名和設定欄（trace、rule、gpu、slow、restore、gpu_GiB、gpu_write、pf_target、policy／gpu_policy 的種類數），沒看任何時間欄。
- 佇列 log 只看了 start／end 列。
