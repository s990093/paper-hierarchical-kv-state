# 第三階段完整 PDF 報告：怎麼產生、結構、narrative 檔、驗收

> 產生器：`code/m16_report_pdf.py`（2026-10-21 寫）。cpu 佇列的 job：`code/m16_queue/jobs/90_report_pdf.sh`（GPU-NEED none；AFTER 由協調者決定）。
> 原則（CLAUDE.md 規則 1、3）：**圖表與表格的每一個數字都在 build 當下從結果檔讀**，程式裡不寫死任何結果；結果檔不存在 → 畫「NOT_MEASURED」佔位、build log 印警告。
> 「解讀、結論、建議」只從 `narrative/<節名>.html` 讀（主 session 撰寫）；不存在 → 紅色虛線框「（待撰寫）」。

---

## 1. 怎麼產生

```bash
cd /mlsteam/workspace/paper-hierarchical-kv-state
source code/m7_env.sh
OMP_NUM_THREADS=4 nice -n 19 taskset -c 0-3 python code/m16_report_pdf.py build          # 約 20 秒
OMP_NUM_THREADS=4 nice -n 19 taskset -c 0-3 python code/m16_report_pdf.py --self-test    # 約 1–2 分鐘，18 項
```

- 只用 CPU：模組載入時裝 import 阻擋器，`torch`／`vllm`／`transformers`／`triton` 一 import 就丟 ImportError（不開 /dev/kfd）。用到的只有 matplotlib、PyMuPDF（`pymupdf`）、fontTools、PIL、markdown-it（venv 裡已有，沒有安裝任何東西）。
- 輸出（預設 `/mlsteam/data/tiara/stage3/report/`，`--out` 可改）：

| 檔 | 內容 |
|:--|:--|
| `stage3_report.pdf` | 報告（A4、頁尾「第 N／M 頁」、PDF 書籤＝第 1–2 層標題、目錄可點） |
| `stage3_report.html` | 同一份內容的 HTML（除錯用；字型用相對路徑 `fonts/`） |
| `figs/figNN.png` | 每張圖（170 dpi）；佔位圖也在這裡 |
| `fonts/tc_regular.ttf`、`tc_bold.ttf` | 這次用到的字元裁出來、轉成 TrueType 的子集字型 |
| `preview/page_NNN.png` | 幾頁的預覽（封面、摘要、目錄、甘特圖、K1、X2、X3、C1b、X4、第 7 章、最後一頁） |
| `build_log.json` | 每張圖／表的狀態（ok／NOT_MEASURED／ERROR）、NOT_MEASURED 清單與原因、讀檔警告、narrative 狀態、每節讀了哪些檔與 run_id、字型與驗收結果、被替換的字元 |
| `history/stage3_report_<UTC>.pdf` | job 每次另存一份（手動 build 不會） |

- 常用參數：`--no-pdf`（只出圖與 HTML）、`--strict`（有警告或 NOT_MEASURED 時 exit 2）、`--preview 12 30`（另外轉這幾頁的 PNG）、`--res／--docs／--narr／--jobs／--fonts`（換來源目錄；self-test 用）。
- exit code：0＝PDF 驗收通過；1＝驗收沒過（出現 fallback 字型或缺頁碼；規則 2：停下來看）；2＝`--strict` 且有警告。

## 2. 結構

| 章 | 內容 | 來源 |
|:--|:--|:--|
| 封面 | 標題、產生時間（UTC／台灣）、平台、結果目錄、git HEAD | 即時 |
| 一頁摘要 | narrative `summary`＋「實驗狀態一覽」（每個實驗的主要結果檔在不在、佇列狀態） | narrative＋檔案與 `/mlsteam/data/tiara/stage3/jobs` |
| 目錄 | 第 1–2 層標題與頁碼（可點） | 排版時算 |
| 1 背景與問題 | 大問題、待遇表、四幕故事、使用者的兩個問題、Intel 四件套對應表；narrative `background` | 事實（plan_v04.html）＋narrative |
| 2 環境與方法 | 硬體與軟體（轉錄 environment_snapshot.md §2–§3）、模型與任務（轉錄 code/m10_tasks.py 的任務表）、共用機器規矩、記錄協定、**實際執行時間軸（甘特圖，讀 runner_timing.log／runner_cpu.log）** | 檔案 |
| 3 文獻與查新 | 研究地圖的一頁結論、L0–L6 總表（卡片數等直接數原檔）、各檔「一頁結論」摘錄、缺口排序；narrative `literature` | lit/、research_map.md |
| 4 實驗結果 | 4.1 K1、4.2 I4、4.3 I1／I2、4.4 C1a、4.5 S1、4.6 X1、4.7 X2＋X2b、4.8 xKV 校準（38c、38k）、4.9 X3（map v1／v2、寫入成本、validate）、4.10 C1b、4.11 X4、4.12 I3、4.13 V0／F2／F1 | 結果檔＋narrative |
| 5 綜合 | 各來源門檻內最省的倍數（並列）、xKV 寫入一次 vs 還原一次、GPU 忙時重建；narrative `synthesis` | 結果檔＋narrative |
| 6 候選題目排名 | narrative `candidates` | narrative |
| 7 限制與誠實聲明 | 模擬 vs 實測、self-review、統計力（CI 半寬即時算）、**失敗清單（從 experiment_log 抽「失敗 N」）**、NEEDS_HUMAN 檔、污染統計；narrative `limitations` | 檔案＋narrative |
| 8 附錄 | results.csv 彙整、結果目錄的 run_id 掃描、重現指令（轉錄 reproduction.md §0–§2）、檔案清單、這份報告怎麼產生的（narrative 狀態、NOT_MEASURED 清單） | 檔案 |

每個實驗節（第 4 章）的固定順序：狀態列 → 要回答的問題 → 作法 → 主要圖表 → 事先寫死的判定與結果（規則原文：能讀檔的直接讀報告 JSON 的 `rule_text`；判定表即時計算或讀報告的 judge／verdict）→ 解讀（narrative）→ 限制 → run_id 與檔案出處（這一節實際讀了哪些檔、檔內的 run_id）。

### 2.1 每張圖讀哪個檔

| 節 | 圖 | 檔 |
|:--|:--|:--|
| 2.5 | 甘特圖 | `/mlsteam/data/tiara/stage3/jobs/runner_{timing,cpu}.log` |
| 4.1 K1 | 低秩誤差 vs rank；跨層 cosine；相鄰 token cosine | `k1_kvstruct/BIG_FILES.md` 指到的 `k1_full/lowrank.csv`、`adjtok.csv`；`k1_kvstruct/xlayer.csv` |
| 4.2 I4 | 各搬法 GiB/s | `i4_copy_batching/copybench_sdma{0,1}.csv` |
| 4.3 I1 | 實測／模擬比；干擾；事先預測 | `i1_concurrent_restore/grid_llama.csv`、`interf_llama.json`、`prediction/prediction_llama.json` |
| 4.4 C1a | recall@16 − 隨機 | `c1a_predictive/c1a_summary.json` |
| 4.5 S1 | 無損比；盈虧 | `s1_lossless/summary.json`、`breakeven.csv` |
| 4.6 X1 | Pareto（家族） | `x_stack/configs_{llama,qwen}.csv`、`x1_report.json` |
| 4.7 X2 | 對 E0 的差 vs 倍數（32K、128K） | `x_stack/v2/x2_feasible.csv`＋`x2_unmapped.csv`（沒有 v2 才讀第一版）、`x_stack/x2_report.json` 的 `by_ctx`、`v2/x2b_report.json` |
| 4.8 | 校準 − 每題 SVD | `x_stack/v2/xlcal_report.json`、`x_stack/v3/xlcal3_report.json` |
| 4.9 X3 | 還原時間（BF16、INT4、模式 X 最好動作；duty 0／0.9） | `x3_time/v2/BIG_FILES.md` 指到的 `x3_full/map_cells.csv`、`v2/map_decision.csv`；寫入成本 `v2/write_cost.csv`；判定 `map_summary.json`（v1、v2）；validate `v2/validate/validate_{cells.csv,summary.json}` |
| 4.10 C1b | 答對率 vs 搬運比 | `c1_selective_restore/c1b_cells.csv`、`c1b_summary.json`、`qtime_*.csv` |
| 4.11 X4 | TF Pareto；生成 vs 倍數；配比、極限、Q-A、多輪、無損表 | `x_stack/x4/x4a_configs_*.csv`、`x4_report.json`、`x4_best_allocation.csv`、`lossless_x4.json` |
| 4.12 I3 | κ 規則對三種基準 | `i3_pause_sim/reread_baselines.csv`、`summary.json` |
| 4.13 | V0 表；F2 TTFT／makespan vs N；F1 TTFT vs 重疊 | `v0_vllm_smoke/v0check_*.json`、`f2_shared_prefix_burst/summary_*.json`、`f1_prefix_reuse/summary_*.json`＋`calib_*.json` |

（路徑都相對 `results/m16_stage3_mi300x/`。）讀不懂的欄位一律不猜：該圖畫「NOT_MEASURED（讀取失敗）」並把例外寫進 `build_log.json` 的 warnings。

## 3. narrative 檔（主 session 撰寫）

位置：`docs/stage3_20261019/report/narrative/<節名>.html`。內容是 **HTML 片段**（`<p>`、`<ul>`、`<ol>`、`<b>`、`<code>`、`<table>`、`<h4>` 都可以），直接放進報告的綠色左線框。

| 節名 | 放在哪裡 | 寫什麼 |
|:--|:--|:--|
| `summary` | 一頁摘要 | 這一輪的結論、候選題目、建議下一步（一頁內） |
| `background` | 1 章末 | 背景補充（可省略；不寫就留待撰寫框） |
| `literature` | 3 章末 | 文獻的整體判讀 |
| `k1`、`i4`、`i1`、`c1a`、`s1`、`x1`、`x2`、`xlcal`、`x3`、`c1b`、`x4`、`i3`、`vllm` | 4.1–4.13 各節的「解讀」 | 該實驗的解讀與結論 |
| `synthesis` | 5 章末 | 壓縮疊加的極限與配比、加速與成本 |
| `candidates` | 6 章 | 第三階段候選題目排名（每個附：問題是否真的存在、前人做到哪、我們的訊號、下一步最該做的一個實驗） |
| `limitations` | 7 章末 | 限制與誠實聲明 |

- 檔案不存在 → 報告放「（待撰寫）」框；只有 `<節名>.draft.html` 也一樣（草稿**不採用**，框裡會提示草稿在哪）。
- `<節名>.draft.html`（這次先建好）：只列事實，數字照 `experiment_log.md` 原文抄、附 run_id；沒結果的寫 NOT_MEASURED。給主 session 參考，改寫成解讀後另存 `<節名>.html`。
- 產生器會移除 narrative 裡的 `<script>`、`<style>`、`<html>／<head>／<body>` 與任何 `font-family`（否則 PDF 會出現 fallback 字型）。字型裡沒有的字元（例如 emoji）會換成 `?` 並記警告；`✗` 換成 `×`、零寬空白刪掉。
- 改了 narrative 只要重跑 `build`（約 20 秒）。

## 4. 驗收

build 會自動做，結果在 `build_log.json` 的 `verify`，也印在最後：

1. **PDF 裡只能有 `Noto Sans CJK TC Regular／Bold`**：逐頁 `get_fonts()`，出現其他字型（特別是 `Droid Sans Fallback`）→ exit 1。
2. **每一頁都有頁碼**：逐頁抽文字，要有「第 N／M 頁」。
3. 書籤數（`toc_entries`）＞0。
4. 圖被 MuPDF 縮小（放不下剩餘空間時它會縮圖）→ 下一輪排版在那張圖前面自動換頁；最後仍被縮小的列在 `render.figs_shrunk`（應為空）。
5. 人工看：`preview/page_*.png`（中文正常、圖表清楚、頁碼正確）。

`--self-test`（CPU、合成資料，18 項）：import 阻擋器；字元收集（含 HTML entity、全形標點、數學符號）；缺字替換；字型子集＋CFF 轉 TrueType（sfnt 0x00010000、有 glyf、沒有 CFF、maxp 1.0、cmap 涵蓋）；PDF 只有我們的字型；**反向對照**（子集外的字一定被抓到 Droid Sans Fallback）；runner log 解析；失敗清單解析；整份報告在「部分合成結果（含還沒跑的 X4、38k、X3 validate、V0、F1、F2，格式照各自產生它的程式）＋其餘不存在」下：沒有讀取錯誤、合成的圖表是 ok、缺的是 NOT_MEASURED、待撰寫框與草稿提示、PDF 驗收通過。輸出在 `/mlsteam/data/tiara/stage3/report/selftest_<UTC>/`，全過就刪掉（`M16_REPORT_SELFTEST_KEEP=1` 保留）。

## 5. 技術作法（踩過的坑）

- 字型：收集整份 HTML 用到的字元（含清單符號 •○ 等 MuPDF 自己產生的）→ fontTools `subset` 把 Regular、Bold 各裁成只含這些字 → CFF 輪廓轉 TrueType（`Cu2QuPen`＋`TTGlyphPen` 建 glyf／loca、刪 CFF／VORG、maxp 1.0、sfntVersion 0x00010000；直接嵌 CFF 子集 MuPDF 讀不到中文）→ CSS `@font-face` 指到 `fonts/*.ttf` → `pymupdf.Story.write_stabilized_with_links`（A4、邊界 50pt）→ 每頁用 `insert_htmlbox` 加頁尾 → `save(garbage=4, deflate=True)`。
- 子集**不留 OpenType layout features**（`layout_features=[]`）：留著的話 MuPDF 會把部分數字換成替代字形，PDF 的文字擷取（ToUnicode）變亂碼（例：`m16` 變成 `mūŰ`）；畫面上看不出來，但複製貼上與搜尋會壞。
- CSS 不用任何 `background-color`：MuPDF 在表格跨頁時會把 `th` 與框的背景畫到別頁（細條狀殘影）。改用框線與文字顏色。
- 圖一律存 PNG（170 dpi），matplotlib 用 `font_manager.addfont` 加 Noto Sans CJK TC；圖寬 490pt、高不超過 690pt。

## 6. 已知限制

- 報告只並列結果檔；不同實驗的題目、題數、長度不同（第 5 章的並列圖已註明「只並列、不直接比」）。
- `results.csv` 由 `code/m16_index.py` 產生，可能比結果檔舊（附錄另有直接掃描結果檔的 run_id 表）；本產生器不重跑 m16_index。
- 第 1 章的背景表是事實摘錄（出自 plan_v04.html），不是讀檔；其他章節的事實文字（問題、作法、限制）寫在產生器裡，不含任何結果數字。
