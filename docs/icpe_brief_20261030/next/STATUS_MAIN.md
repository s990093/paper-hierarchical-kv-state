# STATUS_MAIN（icpe-main，acvlab，分支 `icpe-next-main`，worktree `/ssd6/hungwei/icpe-next/main-ana`）

> 只由 `icpe-main` 寫。格式照 NEXT_PLAN §3b：日期時間／做完什麼（commit、run_id）／卡住什麼（完整錯誤）／需要別的 session 做什麼／需要使用者決定什麼。最新的一節在最上面。
> icpe-main＝**正式的主結果分析 session**：算正式的上限表、照 PREREG 下判定。不跑模擬、不動任何佇列、不碰 `/ssd6/hungwei/icpe-next/data`、不改別人的檔（`next_sim_analyze.py` 原樣使用）。自己的輔助腳本＝`code/icpe/main_*.py`。

## 2026-11-06T09:05+08:00 — PREREG 偏離 7（判定方式的釐清）已寫；版本 A 進 git

**使用者決定**（AskUserQuestion，11/6 09:0x）
- 版本 A：全部 commit，push `icpe-next-main`。
- 判定規則（原始／合成到達各下判定字、Mooncake R1／R2 各自判定、寫入飽和照算另標）：**登記成 PREREG 偏離 7，由 icpe-main 寫**。

**做完什麼**
- **PREREG 偏離 7**（acvlab `date -u` 2026-11-06T00:53:20Z）：只在檔尾加一節（25 行，0 行刪除）。內容＝上面三點、icpe-main 照既有文件的解讀（算判定的元件、工具類的分法、「≥15%」照字面、簡單線索 NOT_MEASURED → 推翻條件「未定」），以及寫的當下已經看過的正式結果。
- **版本 A 的表**：`results/icpe_next/main/verA/`（逐格上限表與 violations 用 xz，解壓後和 run 目錄的原檔 `cmp` 相同；`manifest.json` 有 cmd、context、未壓縮檔的 sha256）。

**需要 h200 做什麼**：偏離 7 在 `icpe-next-main` 的這個 commit。要不要同步進 `icpe-next-a` 由你決定（你同步偏離 5、6 的做法：只取 `PREREG.md` 一個檔）。**下一個偏離編號是 8**。

**需要 icpe-data／icpe-trx 做什麼**：P1 的 block 照常 collect、commit、push 就好；我 fetch 後照 PREREG 偏離 6 算版本 B（主結果＋`_p1`，兩邊同一組 7 個 TTL；偏離 5 的算法並列；`--dead-h 1800`／`3600` 只報告），先核列數。`cw_main_ttl7` 不放（和 P1 重疊）。

**接下來**：等 `{cw,tl,old,bl}_p1` 與 `sa_p1` 全部進 git。

## 2026-11-06T09:00+08:00 — 版本 A（3 個 TTL）完成：對拍 6／6 相同、格數自檢 PASS；判定（版本 A，不是正式值）

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

## 2026-11-06T08:50+08:00 — 開工；使用者對 4 個分析選擇的決定；版本 A（3 個 TTL）開算

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
- 要不要把 2、3 登記成 PREREG 的釐清：→ 已登記為偏離 7（見 09:05 那節）。

**已看過哪些正式結果**（照偏離 5 的慣例照實列出）
- 開工前讀 manifest 時，`stdout_tail` 欄裡的摘要表（我只印了前 600 字元）：
  - `results/icpe_next/b/analysis/sa_main_ttl3/manifest.json`（run `20261106-074822-icpe-trx-analyze-sa_main-ttl3`）：SemiAnalysis N16 s1 的 O-ALL(report)、O-DROP、O-GPU、O-NOW、O-ORD 五列（格數、最大、中位數、≥15%／≥5% 格數）。O-GPU、O-ORD、O-ALL(report) 有 ≥15% 的格子；O-DROP、O-NOW 最大 <3%。
  - `analysis/cw_main_ttl3`、`cw_main_final` 的 manifest：CacheWise N16 s1、s2 的部分元件列，最大都 <1.2%、0 格 ≥5%。
- 讀 sweep 時只看了欄名和設定欄（trace、rule、gpu、slow、restore、gpu_GiB、gpu_write、pf_target、policy／gpu_policy 的種類數），沒看任何時間欄。
- 佇列 log 只看了 start／end 列。
