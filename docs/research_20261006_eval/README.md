# 2026-10-06 評測方法學查證（逐篇評測卡＋複核）

這個資料夾是 `../EVAL_FOUNDATIONS_20261007.md` 的證據附件。
目的：把這個領域「大家怎麼做實驗、參數是什麼、為什麼這樣設計」逐篇整理成同一格式的**評測卡**，
每一格都能追溯到原文的頁碼、章節或程式碼位置，之後再設計不只針對單一方法的 PoC。

流程分兩段：

1. **抽取**：每組由一個子 agent 讀原文（全文，不是摘要），填評測卡。
2. **複核**：另一個沒看過抽取過程的子 agent，對照原文逐格檢查，直接改正錯誤，並在檔尾留下複核紀錄。

---

## 給子 agent 的規則（抽取與複核都適用）

### 硬規則

1. **不准猜，不准編造。** 每一個數字、設定、「某論文用了 X」都要附出處：URL＋位置（§、表、圖、頁；程式碼則是檔名＋行號或函式名）。查不到就寫〔未查證〕。記憶只能當搜尋線索，不能當證據。
2. **讀原文。** 論文優先讀 `arxiv.org/html/<id>`（有 HTML 版時）或 PDF 全文；會議版與 arXiv 版不同時，註明讀的是哪一版。只讀到摘要的，整張卡標〔摘要〕，不得填實驗細節。
3. **證據等級標記**（每格至少一個）：
   - 〔原文〕讀過論文內文
   - 〔程式碼〕讀過原始碼（附 commit 或日期）
   - 〔文件〕官方文件或 README
   - 〔摘要〕只看到摘要
   - 〔二手〕從別的論文轉述（註明是哪一篇）
   - 〔計算〕自己用公開資料算出（附方法）
   - 〔判讀〕自己的推論
   - 〔未查證〕
4. **「為什麼這樣設計」分兩格**：原文自己說的理由（附出處），以及你的判讀（標〔判讀〕）。兩者不可混寫。
5. **引用原文**：以改寫為主。只有定義、措辭本身很關鍵時才直接引用，一次不超過 15 個英文字。
6. **與既有整理比對**：既有檔已查過的事實可引用，但要用原文再確認一次；不一致時以原文為準，寫進該卡的「與既有整理不一致」一格。既有檔：
   - `docs/research_20260924/workloads_eval.md`（35 篇評測矩陣、命中率定義、公開 trace、協定草案 v0）
   - `docs/PAPERS_BY_LEVEL.md`（L0–L5 論文說明）
   - 使用者寄給老師的兩份 PDF 的文字版（見下方「使用者文件」）
7. **只寫自己負責的輸出檔**，不改 repo 裡其他檔案。暫存檔放 scratchpad。
8. 寫繁體中文；術語可以用英文。

### 使用者文件（要核對它們對各論文的描述）

- `RESEARCH_INTRO_20261006`（寄給老師的研究介紹）文字版：`<SCRATCH>/intro.txt`
- `SOTA_ANALYSIS_20261006`（附檔）文字版：`<SCRATCH>/sota.txt`
- Markdown 原稿：`.claude/worktrees/llm-long-context-inference-27746c/docs/RESEARCH_INTRO_20261006.md`、`SOTA_ANALYSIS_20261006.md`

`<SCRATCH>` ＝ `/private/tmp/claude-501/-Users-hungwei-Desktop-Proj-paper-hierarchical-kv-state--claude-worktrees-kv-cache-long-context-research-670052/3b14943d-0ed2-407a-9821-85ac0d2fb19e/scratchpad`

### 工具

`WebSearch`、`WebFetch` 是 deferred tools，先用 `ToolSearch`（query `select:WebSearch,WebFetch`）載入。
PDF 可以用 `curl -L -o <SCRATCH>/<你的資料夾>/x.pdf <url>` 下載後 `pdftotext -layout` 轉文字（本機有 `/opt/homebrew/bin/pdftotext`），頁碼以 PDF 頁為準。
程式碼讀 `raw.githubusercontent.com`，記下 commit SHA 或查證日期。

---

## 評測卡格式（每篇一張）

```markdown
### <簡稱>：<完整標題>（<venue 年>；arXiv <id>）

- **讀了什麼**：〔全文／部分／摘要〕<版本>，<URL>，查證 2026-10-06
- **一句話**：它解什麼問題（40 字內）
- **評測要證明的主張**：原文的主要 claim（改寫）

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | 名稱、大小、注意力類型（MHA/GQA/MQA/MLA）、權重與 KV 的 dtype | |
| 硬體 | GPU、數量、DRAM、SSD／網路、互連 | |
| 軟體與版本 | 引擎與版本、相依套件、自己改了什麼 | |
| 資料／負載 | 名稱、子集、筆數、取樣與前處理 | |
| 長度 | input／output／context 範圍與單位（token 或字元） | |
| 到達與併發 | Poisson λ、真實時間戳、固定間隔、batch、併發上限 | |
| 重用結構 | 多輪、共享前綴、RAG chunk、無 | |
| 掃描的自變數 | 請求率、長度、頻寬、預算、快取容量…… | |
| 對手 | 名稱、版本、參數；是否自己重做 | |
| 系統指標 | 原文的定義、percentile、SLO 門檻 | |
| 品質指標 | 任務指標；與誰比 | |
| 主要結果 | 數字＋條件＋頁碼（只列 2–4 個代表性的） | |
| 消融／敏感度／開銷 | 列出有哪些實驗 | |
| 重複與統計 | 重複次數、誤差棒、信賴區間；沒寫就寫「未說明」 | |
| 程式碼／資料 | 是否公開、URL | |
| 設計理由（原文） | 原文明說為什麼這樣設 | |
| 設計理由〔判讀〕 | | |
| 原文沒講清楚的地方 | 重現時必須自己決定的事 | |
| 與既有整理不一致 | workloads_eval／PAPERS_BY_LEVEL／使用者 PDF 的錯誤或過度簡化 | |
| 對本研究的意義〔判讀〕 | 可以沿用的、不可比的、要小心的 | |
```

非論文的項目（benchmark 工具、資料集）用同樣的精神，欄位可以調整，但「出處」欄不可省略。

### 複核者的做法

1. 不看抽取者的推理，只看卡片，**逐格回到原文**確認。
2. 每格判定：✅ 與原文一致／❌ 錯誤（直接改正，並在該格加〔複核修正〕）／⚠️ 原文找不到（改成〔未查證〕或刪除）。
3. 在檔尾加「## 複核紀錄」：複核者、日期、每張卡檢查了幾格、❌ 與 ⚠️ 各幾格、逐條列出改了什麼。
4. 抽取者漏掉、但對「評測設定」很重要的事實，可以補上並標〔複核補充〕。

---

## 檔案

| 檔案 | 內容 |
|:--|:--|
| `cards/E01_serving_classic.md` | 經典 serving 論文的評測設定＋指標字典 |
| `cards/E02_tools_ecosystem.md` | benchmark 工具（vLLM／SGLang／LMCache／GenAI-Perf／GuideLLM／MLPerf）與 KV 層系統說明書 |
| `cards/E03_restore_recompute.md` | 讀取時還原、重算與載入：Cake、CacheFlow、Pensieve、HCache、KVPR、Bottlenecks、py-kvcache |
| `cards/E04_tiered_systems.md` | 分層儲存系統：Strata、Bidaw、MTDS、LMCache、Mooncake、CachedAttention、Tutti、KVDrive |
| `cards/E05_eviction_policy.md` | 逐出與快取策略：Fancy-eviction、AsymCache、Marconi、AdaptCache、EvicPress、KVCache in the wild、LRB、HALP |
| `cards/E06_learned_eviction.md` | 學習式逐出：LARU、LPC、semantic-aware eviction、KVP、ForesightKV、LookaheadKV、TRIM-KV |
| `cards/E07_compression.md` | 壓縮與量化：StreamingLLM、H2O、SnapKV、PyramidKV、KIVI、KVTuner、KVQuant、CacheGen、CacheBlend |
| `cards/E08_sparse_and_critiques.md` | 稀疏注意力與評測批判：MInference、DuoAttention、Quest、ShadowKV、InfiniGen、Yuan'24、Rethinking'25、Agrawal & Mayer'26、kvpress |
| `cards/E09_quality_benchmarks.md` | 長 context 品質 benchmark：LongBench v1/E/v2、RULER、NIAH、InfiniteBench、SCBench、HELMET 等 |
| `cards/E10_workload_datasets.md` | 負載資料集說明書：ShareGPT、Alpaca、LMSYS、WildChat、BurstGPT、Azure、Mooncake、Bailian、TraceLab 等 |
| `cards/E11_systems_eval_craft.md` | systems 論文的評測手藝：章節結構、量測統計、模擬的可信度、artifact evaluation、審稿人常問 |

狀態與複核結果見各檔檔尾。
