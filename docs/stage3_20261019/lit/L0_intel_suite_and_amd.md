# L0：Intel「KV Cache 智能加速套件」查證，與 AMD MI300X／ROCm 上的對應

> 寫於 2026-10-19。範圍：影片中的四個名稱（KV Shrink／Cascade／Fuse／Infinity）、「agent 一輪約六成時間在搬資料」、「四個難題」；AMD／ROCm 的對應能力；期限 2026-10-23 前在本機做得到的最小實驗。
> **標記**：〔全文〕整份讀過；〔部分〕讀了相關段落；〔摘要〕只讀摘要、或只看到搜尋引擎／擷取工具給的摘錄；〔標題〕只看到標題；〔程式碼 repo@commit 檔名〕；〔本機〕在這台機器上用唯讀指令查到；〔算術〕由已知數字算出；〔判讀〕我的推論；NOT_FOUND＝搜過但沒找到。
> **擷取工具的限制**：中文媒體與部落格頁面是用 WebFetch 讀的。這個工具會先由小模型整理頁面再回傳，所以下面引用的媒體「原文」可能被改寫過，深度一律標〔部分〕。GitHub 上的檔案則是用 GitHub API／raw 下載原檔後直接讀的。
> **引文用字**：中國大陸媒體的引文轉成了繁體字，用語沒有改。
> **規則**：不同硬體的數字只列出，不排名、不互比。Intel 的效能數字全部是廠商或媒體轉述的說法，不是本專案量到的。

---

## 0. 一句話結論

- **四個名稱是真的出現在 Intel 中國區活動上的行銷架構，但只有 KV Shrink 找得到能安裝的東西。** KV Shrink 是 GitHub `intel/intel-accel-for-llm`（vLLM V1 KV connector＋QAT／IAA 壓縮＋DSA 搬運）。KV Cascade、KV Fuse、KV Infinity 只出現在媒體報導，**找不到程式碼、文件或論文**。媒體說「四個元件都已在 GitHub 開源」，但我只找到 KV Shrink。所以後三個**可能是影片或行銷的概念整理，不是可安裝的標準套件**（至少到 2026-10-19 為止）。
- **「一輪約六成時間在搬資料／管上下文／調度儲存」：NOT_FOUND。** 最接近的官方說法是 Intel 2026-08-20 的評論文章：跑程式碼的 agent，GPU 只有 50–60% 的實際時間在做有用的工作。這講的是「GPU 忙碌的比例」，不是「搬資料的比例」，而且文章沒有寫出處。
- **AMD 端最值得量的兩點**（期限前可以做）：
  1. 不靠 QAT 的**無損 KV 壓縮**（GPU 上的 Triton 指數編碼，或 CPU 的 zlib／zstd）：在本專案已量過的三種頻寬（CPU 35.4、SSD 級 6.98、NFS 級 0.331 GiB/s）下划不划算。
  2. DSA 的對應：**零碎 copy 的批次化**。本專案 F1 已量到每次 copy 約 3.5 µs，vLLM 的 CPU 層只跑到 4.1 GiB/s。
  
  第三點是 KV Shrink 的「前 N 層到齊就開始算」，建議期限後再做（§4 E-L4）。

---

## 1. 查證結果

### 1.0 總表

可信度分級：**A**＝有第一手的程式碼或文件可以查；**B**＝Intel 第一手的說法，但沒有資料或測試條件；**C**＝媒體轉述 Intel 的說法（多家說法一致）；**D**＝只在影片裡，或搜不到。

| 名稱 | 影片的說法 | 找到的來源 | 可信度 | 能不能安裝 | 最可能對應的真實技術 |
|:--|:--|:--|:--|:--|:--|
| 套件總名「KV Cache 智能加速套件」 | 四個方向合在一起 | 雲棲大會 Intel 專場（9/23）、Intel Connection 2026（9/22–23 蘇州）的媒體報導；intel.com／intel.cn 搜不到這個名稱 | C | — | 行銷上的總稱 |
| **KV Shrink**（壓縮） | KV 變小、搬得快、讓更多請求跑起來 | **GitHub `intel/intel-accel-for-llm`**（設計文件＋程式碼）；量子位、騰訊新聞、雲棲報導 | **機制 A／效能數字 C** | **可以，但要 Intel 硬體**（QAT／IAA／DSA）＋CUDA 或 XPU GPU；沒有 ROCm 後端 | 分層卸載＋DEFLATE 無損壓縮（BF16 byte-shuffle、只壓指數）＋DSA 批次搬運＋分層管線化載入 |
| **KV Cascade**（聚焦） | 讓模型只處理真正有用的資訊 | 只有媒體：「先請輔助模型做篩選或處理，把相關內容交給主模型」 | C（只有一句描述）；可用性 D | NOT_FOUND | 小模型篩 token：SpecPrefill（ICML'25）；**CacheClip**（作者是 Intel 的 KV Shrink 團隊，輔助模型在 CPU 上跑） |
| **KV Fuse**（復用） | 已存的 KV 真正得到復用 | 只有媒體：「透過快取融合和部分重計算，保留其中能夠復用的工作」 | C；可用性 D | NOT_FOUND | CacheBlend（EuroSys'25）、**CacheClip**（同一團隊） |
| **KV Infinity**（擴展） | 按需載入、預測預取，擴展長上下文 | 只有媒體：「面向持續增長的長任務，通過按需加載和預取」；另一篇把「按需加載」列為**未來技術** | C；可用性 D | NOT_FOUND | InfiniGen、ShadowKV、MagicPIG、RetroInfer（CPU／GPU 稀疏取回）；vLLM `OffloadingConnector`；KV Shrink 自己的分層非同步載入 |
| 「四個難題」（資訊稀疏、儲存不足、擴展受限、復用困難） | 影片 | 所有找到的報導都沒有這四個詞 | D | — | 見 §1.6 我的對應 |

### 1.1 套件整體：誰說的、在哪裡說

- **雲棲大會 Intel 專場**（「智啟 Agent・算築中樞」，2026-09-23）的 CSDN 報導寫道：「配套的 KV Cache 智能加速套件（含 KV Infinity、KV Cascade、KV Fuse、KV Shrink 四個組件）已在 GitHub 開源」。同一篇也說「利用 DSA 加速器進行小數據塊（小於 64KB）的高效搬移」「利用 QAT 對 KV Cache 進行壓縮」〔部分，CSDN 166688978〕。
- **2026 英特爾技術創新與產業生態大會**（Intel Connection 2026，9/22–23 蘇州）：陳葆立（Intel 數據中心集團副總裁、中國區總經理）的演講報導寫「充分調用至強內置的 AMX、QAT 等加速器，推出 KV Cache 智能加速套件」。文中**沒有**拆開講四個元件〔部分，CSDN 166450650〕。
- **量子位**（金磊，2026-09-16）是我**讀得到全文**、而且把四個方向**各寫一句**的報導（引文見 §1.3–1.5）〔部分，微博頭條文章〕。
- **電子工程專輯**（2026-09-26）的搜尋摘錄有一句總述：Infinity＝按需加速與預測預取，Cascade＝篩選並縮短有效上下文，Fuse＝融合獨立的 KV Cache，Shrink＝變小、搬得更快。原頁讀取逾時，只有搜尋摘錄〔摘要〕。
- **Intel 官方網站**：在 intel.com、intel.cn、newsroom.intel.com、community.intel.com 搜「KV Cache 智能加速套件」都沒有結果。2026-10-12 的 OCP 新聞稿（intc.com）也沒有提到這些名稱〔部分〕。
- **GitHub**：用 GitHub API 搜 `kvcascade`、`kv cascade`、`kvfuse`、`kv-fuse`、`kvinfinity`、`kv infinity`、`kv shrink`、`org:intel kv`，**只有 `intel/intel-accel-for-llm` 是 Intel 的**。其他同名 repo 都是個人專案，和 Intel 無關（例如 `dkapur17/KVCascade`，0 星，2026-04 建立）。

**判定**：「四個元件已在 GitHub 開源」這句話**只對 KV Shrink 成立**。Cascade、Fuse、Infinity 在我搜得到的範圍內，是**簡報上的方向名稱，不是可安裝的元件**。

### 1.2 KV Shrink（已查證，可信度 A）

**repo**：<https://github.com/intel/intel-accel-for-llm>〔全文：README、`doc/design/iaxl.md`、`kv-zip.md`、`kvshrink.md`、`benchmark/README.md`；程式碼只看了檔案樹與標頭〕
- GitHub API（2026-10-19 取得）：2026-07-29 建立；最後 push 2026-10-16；HEAD `556c832`；38 個 commit；17 星、7 fork；**沒有 LICENSE 檔**（API 的 license＝null，檔案樹也沒有）。
- 貢獻者：`byang-intel`（Yang, Bin；Intel）24 次、`qiuyuleng1` 5 次、`junzeng1` 3 次、`Yogaraj-Alamenda` 3 次、`cocotyty` 2 次、`rdower` 1 次。

**結構**〔程式碼 intel-accel-for-llm@556c832〕
- `iaxl/`：Intel 的加速層。`KVStore`（用 hash 存取 KV block）→ `KVFlow` → 原生的非同步 `Context`。底下有：
  - 傳輸：`kv_xfer/{cpu,cuda,dsa,rdma,xpu}.cpp`。**沒有 HIP／ROCm 後端。**
  - 壓縮：`kv_zip` 會把一批 block 同時分給 QAT、IAA、CPU zlib 三種 DEFLATE 後端，誰先做完誰接下一個。
  - 快取：分組的 DDR 快取＋LRU＋SQLite 中繼資料＋可選的持久化。
  - 跨節點：`remote_pool`，經 NIXL 走 RDMA。
- `kvshrink/kvshrink_connector.py`：vLLM V1 KV connector。#15（2026-10-10 合併）把基底升到 vLLM v0.29.0。
- 管理用 REST API（`localhost:18700`）：`/v1/cache/status`、`/evict`、`/persist`。媒體說的「冷熱調度 API」〔判讀〕應該就是這組。

**壓縮怎麼做**〔全文 `doc/design/kv-zip.md`、README 的環境變數表〕
- **無損 DEFLATE**（`IAXL_KV_COMPRESSION=1`）。
- `IAXL_KV_DATA_SHUFFLE=1`：先把 BF16 做 byte-shuffle。
- `IAXL_ZIP_EXPONENT_ONLY`：**只壓高位元組（符號＋指數），低位元組原樣存**（#35，2026-10-16 合併，commit 標題：「segmented zip blocks, BF16 exponent-only compression and QAT tuning」）。
- 選配有損的 LSB 截斷（`lossy.cpp`）。
- README：「higher levels barely improve the KV cache compression ratio but make compression much slower」，所以預設 level 1。
- README 說 CPU 壓縮「uses a lot of CPU time」，**預設關閉**。
- **選擇性壓縮**：「latency-sensitive leading layers can bypass compression」。

**搬運怎麼做**〔全文 `iaxl.md`〕
- 「IAXL uses GDRCopy to map GPU memory into a CPU-visible BAR address and submits fragmented H2D or D2H regions as an Intel DSA batch」。
- DSA 不能用時，退回「batched CUDA copies」。
- README 警告：GPU 不支援 P2P DMA 時，不要開 `IAXL_DSA_GD_ENABLE`。

**載入怎麼排**〔全文 `kvshrink.md`〕
- 分層管線：QAT 解壓第 i＋2 層、DSA 搬第 i＋1 層、GPU 算第 i 層。
- 「early-layer promotion」：前 N 層到齊就讓 vLLM 開始 prefill。N 可以依並發數分段設定，例如 `0-3:0,4-6:4,7-:8`。

**效能數字**：README 和設計文件**一個都沒有**。媒體上的數字如下，全部是廠商說法，本專案沒有量過：

| 說法 | 條件（照報導） | 出處 |
|:--|:--|:--|
| TTFT 最高約 5 倍（相對原生 vLLM） | 雙路至強金牌 6554S、2×NVIDIA L20、Qwen3-32B、快取命中率 80%、vLLM 自帶 benchmark | 量子位、騰訊新聞〔部分〕 |
| 相對 LMCache：單路平均 TTFT 129.81 → 114.13 ms（−12.1%）；8 路並發 −4.6% | 雙路至強金牌 6554S、8×H800、Qwen3-32B FP8、命中率 80%、英特爾與道客聯合實驗室 | 量子位、騰訊新聞〔部分〕 |
| QAT 硬體壓縮的整體效能約為 CPU 軟體壓縮的 2 倍；開 QAT 後 TTFT 額外開銷 < 10% | 同第一組 | 量子位〔部分〕 |
| 平均壓縮率 20%+（資料重排前 10%+）；儲存空間減 20–30%；無損 | 沒寫 | 騰訊新聞〔部分〕 |
| 「KV Cache 傳輸帶寬最多可提升 9.66 倍，無損壓縮率提高 1.42 倍，首 Token 延遲最多改善 15 倍」 | **沒寫**。ZOL 說 9.66 倍是用 DSA 把 KV 從顯存拷到系統記憶體；鉅亨網說是「達 CUDA 方案近 10 倍」 | CSDN（雲棲）、ZOL、鉅亨網〔部分〕 |
| 「QAT 引擎實現 50% 以上壓縮率，300GB 可壓至 200GB」 | 沒寫 | 鉅亨網、ZOL〔部分〕 |

- **報導之間互相矛盾**〔算術〕：300 GB → 200 GB 是少了 33%，不是「50% 以上」；這又和騰訊新聞的「平均 20%+」不同。可見媒體的壓縮數字沒有統一的定義，**不能引用**。
- 騰訊新聞把「有損壓縮（50% 以上）」和「按需加載」列為**未來技術**〔部分〕。所以 2026-09 時，這兩項還不在 KV Shrink 裡〔判讀〕。

### 1.3 KV Cascade（只有媒體一句話）

- 量子位原句（經擷取工具）：「KV Cascade 則嘗試先請輔助模型做篩選或處理，把相關內容交給主模型。」〔部分〕
- 鉅亨網：「AMX 加速數據預處理與向量數據庫」〔部分〕。〔判讀〕輔助模型可能放在 Xeon 上用 AMX 跑，但沒有任何來源明講。
- **對應的真實技術**：
  - **CacheClip**（arXiv 2510.10129；v1 2025-10-11、v2 2026-05-21）。作者 Bin Yang、Qiuyu Leng、Jun Zeng、Zhenhua Wu。摘要：用輔助模型挑出要重算的 token，並且「offloads auxiliary model inference to idle CPU resources」〔摘要〕。
    - 作者和 KV Shrink repo 的前三名貢獻者（`byang-intel`＝Yang, Bin／Intel、`qiuyuleng1`、`junzeng1`）**名字對得上**〔本機 GitHub API〕。〔判讀〕Cascade 或 Fuse 很可能出自這個團隊的這條研究線，但**沒有來源確認**。
    - CacheClip 的程式碼：NOT_FOUND（GitHub 搜 `CacheClip` 只有無關的 repo）。
  - **SpecPrefill**（arXiv 2502.02789，ICML 2025）：用小模型估 token 重要性，只把挑出的 token 交給大模型〔摘要〕。
- **本專案的關聯**：Tiara 的動作空間裡沒有「丟掉 token 不算」這種有損的上下文刪減。要做就必須量品質（ε），期限前做不完（§4）。

### 1.4 KV Fuse（只有媒體一句話）

- 量子位原句（經擷取工具）：「KV Fuse 嘗試通過緩存融合和部分重計算，保留其中能夠復用的工作。」前文說明：獨立產生的幾份 KV「通常不能直接拼接，因為模型處理文件時會受前文和位置影響」〔部分〕。
- **對應**：
  - **CacheBlend**（EuroSys'25，arXiv 2405.16444）：本專案已有卡片，見 `docs/PAPERS_BY_LEVEL.md`、`docs/PAPER_GUIDE.md`〔全文，前次〕。
  - **CacheClip**（同 §1.3，摘要裡把 CacheBlend、APE 當對照組）。
  - `qiuyuleng1` fork 了 TurboRAG（RAG 的預算 KV 拼接）〔本機 GitHub API〕。這只是旁證〔判讀〕。
  - 另有 FusionRAG（arXiv 2601.12904）〔標題〕。
- **本專案的關聯**：本專案在 vLLM 的前綴語意下，重算只能從最前面連續做（`docs/precision_20261013/02_papers.md` §2）。Fuse 型的「中段重用＋部分重算」不在現在的動作空間裡。

### 1.5 KV Infinity（只有媒體一句話）

- 量子位原句（經擷取工具）：「KV Infinity 面向持續增長的長任務，通過按需加載和預取，嘗試緩解緩存必須全部常駐 HBM 的壓力，讓系統能夠利用顯存之外的資源。」同一篇說「具體能擴展到什麼程度，仍取決於訪問方式和傳輸效率」〔部分〕。
- 騰訊新聞（Intel Connection 預告）把「按需加載」列為 KV Shrink 的**未來技術**〔部分〕。所以 Infinity 很可能還沒有發布的實作〔判讀〕。
- **對應**：
  - InfiniGen（OSDI'24，預演下一層 attention 再預取）
  - ShadowKV（ICML'25，arXiv 2410.21465）
  - 以上兩篇本專案都有卡片〔全文，前次〕。
  - MagicPIG（ICLR'25，arXiv 2410.16179，CPU 上做 LSH 取樣的 attention）〔摘要〕
  - RetroInfer（arXiv 2505.02922，wave index＋wave buffer 跨 GPU／CPU）〔摘要〕
  - ScoutAttention（arXiv 2603.27138，提前一層在 CPU 預算 attention）〔摘要〕
  - 不改模型的版本：vLLM `OffloadingConnector`＋`TieringOffloadingSpec`（CPU 主層＋檔案系統層）〔部分，vLLM 文件〕，以及 KV Shrink 自己的分層非同步載入（§1.2）。

### 1.6 「四個難題」

- **NOT_FOUND**：搜到的所有報導都沒有「信息稀疏、存儲不足、擴展受限、復用困難」這組說法。鉅亨網講的是「兩點」挑戰（KV Cache 與沙盒隔離）〔部分〕。投資界的報導講的是「三重考驗」，而且指的是軟體生態與成本，不是 KV〔部分〕。
- 〔判讀〕影片很可能自己把四個難題各對到一個元件：
  - 資訊稀疏 → Cascade
  - 儲存不足 → Shrink
  - 擴展受限 → Infinity
  - 復用困難 → Fuse
  
  這個對應沒有出處，**不能當成 Intel 的說法引用**。

### 1.7 搜了什麼、在哪裡搜

- **中文關鍵字**：
  - 「英特尔 KV Cache 智能加速套件 KV Shrink KV Cascade KV Fuse KV Infinity」
  - 「"KV Infinity" 英特尔 按需加载 预取 长上下文」
  - 「"KV Fuse" 英特尔 缓存融合 部分重计算 复用」
  - 「英特尔 KV Cache 四个难题 信息稀疏 存储不足 扩展受限 复用困难」
  - 「智能体 一轮 六成时间 数据搬运 上下文管理 存储 英特尔」
  - 「"60%" 智能体 时间 数据搬运 … 陈葆立」
  - 「"约60%" OR "六成" 智能体 任务 时间 …」
  - 「intel.cn 新闻 英特尔技术创新与产业生态大会 2026 KV Cache」
  - 「KV Cache 智能加速套件 英特尔」（限定 intel.com／intel.cn 網域）
- **英文關鍵字**：
  - `Intel "KV Shrink" "KV Cascade" "KV Fuse" "KV Infinity"`
  - `github intel "KV Shrink" QAT vLLM`
  - `"KV Infinity" Intel github …`
  - `"KV Cascade" OR "KVCascade" Intel …`
  - `Intel agentic AI "60%" time spent data movement …`
  - `Intel KVShrink QAT DSA KV cache connector vLLM blog`
  - `LMCache Intel QAT compression backend OR opea-project …`
  - `Hot Chips 2026 Intel Xeon 6+ KV cache offload QAT DSA`
  - `ROCm GPU lossless compression … hipCOMP`
  - `hipFile …`
  - `MoRI …`
  - `AITER …`
  - `HSA_ENABLE_SDMA …`
  - `AMD EPYC PTDMA …`
  - `AMD EPYC … QAT equivalent`
- **GitHub API**：上面那組 repo 搜尋；`intel/intel-accel-for-llm` 的 commits、tree、contributors；三位貢獻者的 repo 清單；`CacheClip`。
- **讀過的頁面**：
  - 騰訊新聞 ×2（20260915、20260904）
  - ZOL（亂碼，只能部分讀）
  - CSDN ×2（166688978、166450650）
  - 量子位的微博頭條文章
  - 新浪、鉅亨網、投資界
  - 美通社（憶聯 G3.5）
  - intc.com 的 OCP 新聞稿
  - Intel 評論文章（newsroom 原址已 301 到 404；改讀 SemiWiki 轉貼）
- **讀不到的**：
  - 電子工程專輯（逾時）
  - OFweek 維科號（404）
  - AMD MI300X 產品頁與 datasheet PDF（逾時；只用搜尋摘錄）
  - Springer 上的 AMD GPU 壓縮論文（要登入）
- **沒搜到 Intel 對應物的地方**：opea-project、LMCache（沒有 QAT／DSA 後端）、Hot Chips 2026（Intel 講 Diamond Rapids／Crescent Island，沒講 KV 卸載）。

---

## 2. 「AI agent 一輪約六成時間花在資料搬運、上下文管理和儲存調配」

**結論：NOT_FOUND。** 中文和英文、Intel 官網和媒體都搜過，沒有一個來源有「六成／60%」配上「資料搬運、上下文管理、儲存調配」的說法。不要引用。

最接近的三個說法（各自量的東西不同，都不能等同於「六成時間在搬資料」）：

| 來源 | 原文數字 | 量的是什麼 | 量測條件 | 深度 |
|:--|:--|:--|:--|:--|
| Intel 評論〈Count the Work Done. Not Just the Cores.〉（Karin Eibschitz Segal，2026-08-20；SemiWiki 2026-08-21 轉貼） | 「the same research estimates that GPUs perform useful work during only 50% to 60% of wall-clock time」（跑程式碼的 agent） | **GPU 有在做有用工作的時間比例**。不是搬資料的比例；反過來算是 40–50% 閒置〔算術〕 | 文中只說「a recent public analysis using 739 anonymized Claude Code conversations」，**沒寫研究名稱與連結**；同一份分析說 CPU 端處理＋等待從單一請求的 <1% 升到 32 並發時的「as much as 15%」 | 〔部分〕（newsroom 原址 404） |
| arXiv 2511.00739 v1（Raj, Wang, Krishna，2025-11-01） | 「Tool processing on CPUs can take up to 90.6% of the total latency.」 | 工具呼叫在 CPU 上的處理時間 | 摘要沒寫硬體；v3（2026-04-16，標題改為〈Towards Understanding, Analyzing, and Optimizing Agentic AI Execution: A CPU-Centric Perspective〉）的摘要**不再寫**這個百分比 | 〔摘要〕 |
| arXiv 2605.26297〈Agentic AI Workload Characteristics〉（IISWC 2026） | 摘要沒有任何百分比；只說「with effective context caching, most input tokens are reused across turns, making execution decode-dominated」 | — | 5 個 agent benchmark、Gemma／Qwen | 〔摘要〕 |

- 旁證〔判讀〕：「739 anonymized Claude Code」這組資料，和 LMCache 在 MI300X 上 benchmark 用的「739 anonymized Claude Code agentic traces from the kv-cache-tester toolkit by callanjfox/WEKA」數量相同。Intel 引的很可能是這個公開資料集，但 Intel 沒寫，無法確認。
- 〔判讀〕影片的「六成」最可能是把「GPU 只有五到六成時間有用」**倒過來講**，或是口頭整理。真要在論文裡用，只能寫成「廠商宣稱」，或者乾脆不用。本專案的主張（κ 跨硬體變動）也不需要這個數字。

---

## 3. AMD／ROCm 上的對應能力

### 3.1 這台機器的實際狀態（2026-10-19，唯讀指令）

| 項目 | 實況 | 出處 |
|:--|:--|:--|
| CPU | AMD EPYC 9684X 96-Core；**cgroup 配額 32 CPU**（`cpu.max`＝3200000/100000） | 〔本機〕`lscpu`、`/sys/fs/cgroup/cpu.max` |
| CPU 指令集 | 有 AVX-512 F／BF16／VNNI／VBMI／VBMI2；**沒有 AMX** | 〔本機〕`/proc/cpuinfo` |
| 記憶體 | cgroup 上限 466,877,906,944 B（＝434.8 GiB）；`/dev/shm` 179 G。**任務敘述的「2.2 TB DRAM」是主機的 MemTotal，容器用不到** | 〔本機〕；`docs/MI300X_MLSTEAM.md` |
| 本地 NVMe | **沒有**（`/dev/nvme*` 不存在）。根目錄是 overlay 3.0 T（sda）。**任務敘述的「local NVMe」與實況不符** | 〔本機〕`df -hT`；`docs/research_20261010_directions/D7_os.md` |
| NFS | `/mlsteam/workspace`、`/mlsteam/data/tiara`（nfs4，950 G） | 〔本機〕 |
| ROCm | 7.2.2 | 〔本機〕`/opt/rocm/.info/version` |
| Python 套件（`venv/tiara-v028`） | torch 2.12.1+rocm7.2；triton 3.8.0＋triton_rocm 3.7.1；vLLM 0.28.0+rocm722；zstandard 0.25.0；標準庫 zlib 1.3 | 〔本機〕site-packages 清單 |
| 沒有的 | lz4、hipCOMP、hipFile、LMCache、AITER（兩個 venv 都沒有；`/opt/rocm/lib` 也沒有 hipfile／comp） | 〔本機〕 |

### 3.2 Intel 元件 ↔ AMD 的對應

| Intel（KV Shrink 用到的） | 作用 | AMD MI300X＋EPYC 上的對應 | 這台能不能用 |
|:--|:--|:--|:--|
| **QAT**（Xeon 內建的 DEFLATE 硬體） | 卸載時壓縮／解壓，不占 CPU 核心 | **EPYC 沒有內建對應物**。AMD 自己的白皮書是把 EPYC 9004 配上 MaxLinear Panther III 外接卡〔摘要〕。在 AMD 上只能用 (a) CPU 軟體壓縮（zlib／zstd）或 (b) GPU 壓縮 | (a) 可以：32 核、zlib／zstandard 都已裝；(b) 見下三列 |
| （GPU 壓縮，Intel 沒用） | — | **hipCOMP-core**（ROCm 官方；nvCOMP branch-2.2 的移植；README 寫「early-access software technology preview」「experimental and not yet performance-optimized」；LZ4／Snappy／Cascaded／RLE 可壓可解；Zstd、Deflate 只能解；Bitcomp／ANS／GDeflate 不支援；測過 gfx942、ROCm 7.1.1–7.2.3）〔部分〕 | 沒裝，要從原始碼建 |
| | | **hipANS**（dietGPU 的 HIP 版；有 bf16 的 float codec；README 只寫 MI100／MI210 測過，宣稱 50–300 GB/s）〔部分〕 | 沒裝；MI300X 沒測過 |
| | | **SplitZip**（Notre Dame；arXiv 2605.01708；README 寫 NeurIPS 2026）：**純 Triton**，沒有 CUDA 專屬程式碼〔程式碼 SplitZip@42c7431 `codec_gpu.py`〕。README 寫「requires a CUDA-capable GPU」，論文數字是 H200 上量的 | **〔判讀〕可能不用改就能在 ROCm Triton 上跑，但沒驗證**（§4 E-L1 的第 0 步） |
| **IAA** | 另一種 DEFLATE 硬體 | 沒有 | — |
| **DSA**＋GDRCopy | 把零碎的 H2D／D2H 片段當成一批 DSA 描述子送出 | (a) GPU 端的 **SDMA 引擎**（`HSA_ENABLE_SDMA`，預設 1）或 blit kernel（設成 0）。ROCm 文件說 blit「low latency (typically < 5us) and can achieve approximately 80% of the bandwidth of the DMA copy engine」〔摘要〕。(b) 一個 gather kernel 一次搬完。(c) ROCm PR #11614 把 `hipMemcpy3DBatchAsync` 合成一次 SDMA 送出：**還沒合併**；作者在 MI355X、N＝512 量到 64 KB 時 10.88 → 1.44 µs/op〔部分〕。(d) 主機端的 PTDMA／AE4DMA：官方說明是為 NTB 設計，「not intended for general purpose peripheral DMA」〔摘要〕 | (a)(b) 可以；(c) 7.2.2 沒有；(d) 容器內不可用〔判讀〕 |
| **AMX** | 輔助模型、前處理 | 沒有 AMX；有 AVX-512 BF16／VNNI | CPU 推論可以，但慢多少：NOT_MEASURED |
| vLLM V1 connector | 接進 serving | vLLM `OffloadingConnector`：文件寫「currently supports CUDA, ROCm, and XPU only」，GPU↔CPU 走 DMA、可以設 `blocks_per_chunk`〔部分〕。本專案在 MI300X 上已經在用 | 可以（`venv/tiara-v028`） |
| 分層快取 | DDR／SSD／遠端 | **LMCache 的 ROCm 版**（`BUILD_WITH_HIP=1`；LMCache 部落格在 2×MI300X 上跑過）〔部分〕。**AMD Infinity Context（AIC）**：LMCache 管 block，下面接可 RDMA 的網路儲存；支援 MI300X／MI350；2026-09 Tech Preview；文章沒有提到壓縮〔部分〕 | LMCache 沒裝；AIC 要 RDMA NAS：不可用 |
| 遠端池（RDMA） | 跨節點共用 KV | **MoRI／MORI-IO**（ROCm/mori；KV 傳輸；RDMA／XGMI／TCP；vLLM 有 `MoRIIOConnector`，配方以 MI355X P/D 為例）〔摘要〕 | 單一 pod，用不到 |
| GPUDirect Storage 類 | GPU↔NVMe 直通 | **hipFile／AMD Infinity Storage**：AMD 說 ROCm 7.14 GA；ROCm 10.1（2026-10-05）加了非同步快速路徑〔部分〕。本專案 D7 已查過條件：本地 NVMe＋ext4／xfs＋P2PDMA | **不可用**（沒有 NVMe、ROCm 7.2.2） |
| 互連 | — | PCIe Gen5 x16（datasheet 寫 128 GB/s）〔摘要〕；xGMI 每顆 7 條，AMD 部落格寫「rated up to 64 GB/s for each point-to-point link」、實際約 45–48 GB/s（8 卡平台的 TransferBench）〔部分〕 | 本 pod 只看得到 2 張卡，xGMI 全互連的數字不適用〔判讀〕 |

### 3.3 本專案已經量過的 MI300X 搬運數字（給 §4 用）

| 量 | 值 | 出處 |
|:--|:--|:--|
| CPU 層（pinned、真實 H2D，×0.9 降額） | 讀 35.4159、寫 35.5482 GiB/s | `results/m7_write_policy_mi300x/tier_params.json`（run `20261008-130257-m7-c0`、`20261008-130438-m7-c2`） |
| 本地 SSD 級 | 讀 6.981、寫 1.911 GiB/s | 同上 |
| NFS 級 | 讀 0.331、寫 0.601 GiB/s | 同上 |
| 裸 pinned H2D | 07_report §7 的表寫「50 GiB/s（裸）」，正文寫「53 GiB/s」（**同一份文件裡兩個數字**，沒有修正） | `docs/phase1_20261008/07_report.md` §7 |
| vLLM 0.28 `OffloadingConnector` 的 CPU→GPU 實得頻寬 | 8K／16K／32K：4.02／4.09／4.11 GiB/s | run `20261008-141324-m7-c6-vllm`（07_report §7） |
| 每次 copy 的送出成本 | ≈ 3.5 µs／次（含合併用的 numpy 時間）〔算術〕；合併 copy 讓等載入的空轉 21.5 → 11.0 s | `docs/research_20261010_followup/F1_offload_slowdown.md` §4.6（run `20261010-112119-f1-s1-rocmfixcoal`） |
| 有損 FP8 在各層的還原時間（E0 → E1） | CPU 層，閒：Llama 95.4 → 71.7 ms、Qwen 52.2 → 54.9 ms（Qwen 輸）；限速層 E1 快 22.5–39.2% | `docs/precision_20261013/08_report_mi300x.md` |

AMD 已公開、和 KV 壓縮或卸載有關的數字（**都不是 MI300X 上的無損壓縮，只列不比**）：
- ROCm 部落格〈4-bit KV Caching in LMCache〉（2026-08-28）：**MI355X**、**有損**（TurboQuant）。同記憶體下 goodput 385 vs 189 個請求，p95 TTFT 53 vs 140 s，D2H＋H2D 的 copy 時間 50.2 s（4.2%）vs 69.6 s（5.8%）〔部分〕。
- LMCache 部落格（2026-05-12）：**2×MI300X**、vLLM 0.19.0、MiniMax-M2.5 FP8、64 GB CPU L2、739 條 agent trace。壓力階段 TTFT 平均 34.59 s（LMCache）vs 102.17 s（只用 HBM 前綴快取）。合成命中率測試裡 LMCache **慢 10–17%**〔部分〕。

### 3.4 MI300X 上能做、不能做

- **能做**：
  - CPU 層卸載（vLLM connector、自寫 harness）
  - 本地 SSD 級、NFS 級的實測與限速模擬
  - GPU 端 Triton 寫的 codec
  - CPU 端 zlib／zstd（32 核內）
  - SDMA／blit／gather kernel 的零碎 copy 比較
  - FP8／INT4 有損（本專案已有）
- **不能做（這台）**：
  - QAT／IAA／DSA 這類 CPU 端硬體加速：EPYC 沒有
  - hipFile 直通：沒有 NVMe，ROCm 也太舊
  - AIC、MoRI：沒有 RDMA 儲存，也沒有多節點
  - AMX 上跑輔助模型
  - 能耗：CLAUDE.md 已禁止在平台 A 下能耗結論；本機 RAPL 讀不到（D8）
- **要安裝才能做**：hipCOMP、hipANS、LMCache。**這次沒有安裝**。要裝的話，要寫成 `/mlsteam/workspace` 下的腳本（MI300X 手冊 §0）。

---

## 4. 期限（2026-10-23）前能做的最小實驗

**大前提**
- 今天是 10/19，只剩約 4 天。機器正在跑計時量測，所以：
  - GPU 實驗要等空卡，並用 `gpu_guard`。
  - CPU 壓縮實驗**不能和計時量測同時跑**，會污染對方的時間欄。
- 每個 run 都照 CLAUDE.md §4 留 run_id。

**為什麼先做「無損」**：無損壓縮不改 KV 的位元組，ε＝0，所以不用跑品質。這和 CLAUDE.md「CPU 與 SSD 兩階是無損的位元組搬移」的定義一致：加上壓縮後，只是同一個動作的成本常數變了。〔判讀〕這正好是 κ 的論點：Intel 用硬體（QAT、DSA）改變 B 與每次 copy 的成本；AMD 上要改，只能靠軟體。

### 4.0 先算損益兩平（〔算術〕，全部從 §3.3 的實測頻寬算）

符號：
- S：原始 BF16 位元組數
- B：層的讀頻寬
- r：壓縮比（原始 ÷ 壓縮後）
- D：解壓吞吐（以原始位元組計）

兩種做法：
- **不管線化**（先搬完再解壓）：要 S/(rB)＋S/D < S/B，也就是 **D > B·r/(r−1)**。
- **管線化**（邊搬邊解，像 KV Shrink 的分層管線）：要 **D > B**。
- 不管哪一種，最多只能省 **1 − 1/r** 的搬運時間。

r 的上限從哪來：
- SplitZip 的格式是每個元素 4 bit 指數碼＋8 bit 符號與尾數，再加例外串流，所以 **r ≤ 16/12 ＝ 1.333**〔程式碼 SplitZip@42c7431 `encode()`：`pk` n/2 bytes、`sm` n bytes〕。
- Intel 媒體說平均壓 20%+，約等於 r ≈ 1.25。

| 層（讀頻寬） | 最多省（r＝4/3） | 不管線化要的 D（r＝4/3） | 不管線化要的 D（r＝1.25） | 管線化要的 D |
|:--|--:|--:|--:|--:|
| CPU 35.4 GiB/s | 25% | 141.6 GiB/s | 177.0 GiB/s | 35.4 GiB/s |
| SSD 級 6.98 GiB/s | 25% | 27.9 GiB/s | 34.9 GiB/s | 6.98 GiB/s |
| NFS 級 0.331 GiB/s | 25% | 1.32 GiB/s | 1.66 GiB/s | 0.331 GiB/s |

寫入端（壓縮）同理：不管線化時，SSD 寫 1.91 GiB/s 要 E > 7.64 GiB/s，NFS 寫 0.601 GiB/s 要 E > 2.40 GiB/s（r＝4/3）。

〔判讀〕
- CPU 層：解壓幾乎一定賺不回來。有損 FP8 在 CPU 層已經被反量化吃掉（Qwen 輸），無損的 r 又只有約 1.3。
- SSD 級：要看 GPU codec 夠不夠快。
- NFS 級：CPU 多執行緒的 zlib／zstd 就可能夠。
- **E-L1／E-L2 就是要用實測取代這三句猜測。**

### E-L1：GPU 端無損 BF16 指數編碼（對應 KV Shrink 的壓縮，但不靠 QAT）

- **假說 H1**：在 MI300X 上，SplitZip 的 Triton codec（不改或只改裝置字串）跑真實 Llama／Qwen 的 KV：
  - r ≥ 1.2；
  - 解壓 D ≥ 28 GiB/s（過 SSD 級不管線化的門檻）。
  
  所以 SSD 級與 NFS 級限速層上，還原時間省 ≥ 15%；CPU 層省 ≤ 5% 或變慢。
- **第 0 步（可行性，約 15 分鐘 GPU）**：
  - 用 `--synthetic --calibrate-on-input` 冒煙，確認 gfx942 上 Triton 編得過，而且 `torch.equal(x.view(int16), decoded.view(int16))`。
  - **編不過或不相等 → 記下完整錯誤，停下來回報**（CLAUDE.md 規則 2）。不要自己改 kernel 硬跑；改走 E-L2，並把 hipANS 列為「要安裝才能試」。
- **對照組**：
  - E0（BF16 原樣）
  - 本專案已有的 FP8 E1（有損，只當參考，不和無損比品質）
  - 指數熵的理論上限（只做統計，不編碼）
- **量什麼**：
  - 每個（模型, 層, K／V）的 r，例外比例
  - 編碼／解碼的 GB/s：GPU 計時、閒／忙兩種狀態
  - 逐位元比對（每個 block 都比）
  - codebook 用另一組 prompt 校準（照 SplitZip README，不在測試資料上校準）
- **接進還原**：在 `m10_codec.py` 的格式清單加一個無損格式。`Enc.nbytes`＝壓縮後的位元組，讓 `m10_tier2.py` 的限速器照它計時。在 cpu／ssd（6.98）／nfs（0.331）三層各量 E0 與「無損」。
- **事先寫死的判準**：
  - 任何 block 位元組不同 → 整個實驗作廢。
  - SSD 級、NFS 級的還原時間中位數差 > δ\*（沿用 09 §5 的雜訊門檻）而且配對 CI 不含 0 → 「無損壓縮在該層划算」。
  - CPU 層同樣的判準反過來寫。
- **預估工作量**〔估計，非量測〕：寫 codec 包裝與接線約 0.5 天；GPU 約 1–2 小時（冒煙＋microbench＋兩個模型 × 三層 × 閒／忙的還原）。

### E-L2：CPU 端無損壓縮（量出「沒有 QAT 要付的代價」）

- **假說 H2**：在 32 核配額內，對 BF16 KV 做 byte-shuffle、只壓指數位元組（照 IAXL 的作法），zlib level 1／zstd level 1 的 r 約 1.2–1.3。但就算開 16 執行緒，解壓 D 也 < 28 GiB/s，所以只在 NFS 級划算。
- **對照組**：
  - 不 shuffle vs shuffle vs 只壓指數
  - zlib（和 QAT 同樣是 DEFLATE 格式）vs zstd
  - 執行緒 1／4／16
- **量什麼**：
  - r
  - 單執行緒與 N 執行緒的吞吐
  - **每 GiB 花掉幾 CPU 秒**。這就是 QAT 替 Intel 省下、AMD 主機要自己付的成本，直接對應 KV Shrink 宣稱的「不占 CPU」。
- **資料**：用 `m10_kv_stats.py` 同一條路徑產生的真實 KV。
  - 先在 GPU 上存一份 BF16 KV 到 `/mlsteam/data/tiara/runs/<run_id>/`（大檔不進 git）。
  - 之後的壓縮全部在 CPU 上離線做。
- **注意**：
  - **必須和任何計時量測錯開**，因為會搶 CPU。
  - 只用已裝的 `zlib`／`zstandard`，不裝 lz4。
- **預估工作量**〔估計〕：0.5 天；純 CPU 約 1 小時。

### E-L3：零碎 copy 的批次化（對應 DSA）

- **為什麼值得做**：本專案已量到 vLLM CPU 層只有 4.0–4.1 GiB/s，比 pinned H2D（35.4 降額／50–53 裸）慢 9–12 倍。F1 也查出主因是「每次 copy 的固定成本 × copy 次數」（≈3.5 µs／次），並建議「gather kernel、更好的合併、或真正批次化的 memcpy」（F1 §4.6、§6）。這正好是 Intel 用 DSA 處理的問題（媒體寫「小於 64KB 的小數據塊」），也是 ROCm PR #11614 要修的問題。
- **假說 H3**：在 ROCm 7.2.2、MI300X 上，用 vLLM 的片段大小（若 block_size＝16：Llama 每層每個 K 或 V 是 16×8×128×2 B＝32 KiB，Qwen 是 16 KiB〔算術〕），以下兩種做法搬同一組片段：
  - (b) 一個 gather kernel（直接讀映射進來的 pinned 主機記憶體）
  - (c) `HSA_ENABLE_SDMA=0` 的 blit
  
  至少比 (a) 逐片 `hipMemcpyAsync`（SDMA）快 3 倍。
- **對照組**：
  - 同樣總位元組的一次連續 copy（上限）
  - (a)／(b)／(c)
  - 片段大小掃 4 KiB–1 MiB
- **量什麼**：
  - GiB/s
  - µs／片段
  - GPU 忙時的干擾：blit 會占 CU，SDMA 不會，所以兩種都要量
- **預估工作量**〔估計〕：寫 2–3 小時；GPU 約 1 小時。
- **對論文的意義**〔判讀〕：CPU 層的成本常數會因軟體路徑而差約 10 倍。κ 的跨硬體論點要說清楚：量到的是「硬體＋軟體路徑」，不只是硬體。

### E-L4（選做，期限前不建議）：前 N 層到齊就開始算（KV Shrink 的 early-layer promotion）

- 先查本專案的還原器（`m7_restore_harness.Restorer`、Cake 雙向）是不是已經做到逐層重疊。已經有了就不用做。
- KV Cascade（輔助模型篩選）、KV Fuse（中段融合＋部分重算）都是**有損**的，要跑品質才能下結論；Infinity 需要稀疏 attention 的 kernel。這三項**期限前都做不完**，不排。

### 建議排程

| 日 | 事 | 資源 |
|:--|:--|:--|
| 10/19–20 | 存一份真實 KV（GPU 約 15 分鐘）→ 指數熵統計＋E-L2（CPU 離線，等計時量測空檔） | 1 張空卡＋CPU 空檔 |
| 10/20–21 | E-L1 第 0 步冒煙；過了才做 microbench | 1 張空卡 |
| 10/21–22 | E-L3 microbench；E-L1 接進 `m10_tier2` 的三層還原 | 1 張空卡 |
| 10/22–23 | 分析、寫 RUNLOG；判準沒過就照實寫 NOT_GO | — |

---

## 5. 參考連結

### 5.1 Intel 端

| # | 連結 | 內容 | 深度 |
|:--|:--|:--|:--|
| I1 | <https://github.com/intel/intel-accel-for-llm>（HEAD `556c832`，2026-10-16） | KV Shrink／IAXL 原始碼 | 〔全文〕英文 README、`doc/design/{iaxl,kv-zip,kvshrink}.md`、`benchmark/README.md`；中文 README 與 `remote-pool.md` 下載了但沒讀；程式碼只看檔案樹 |
| I2 | <https://github.com/intel/intel-accel-for-llm/pull/15>、<https://github.com/intel/intel-accel-for-llm/pull/20> | vLLM v0.29.0 升級；CPU 推論 | 〔摘要〕（合併日期取自 commit log） |
| I3 | <https://news.qq.com/rain/a/20260915A0CVB300> | 騰訊新聞，Intel Connection 預告，KV Shrink 數字 | 〔部分〕 |
| I4 | <https://news.qq.com/rain/a/20260904A0DHFL00> | IT 時代網：KV Shrink 發布（與道客聯合實驗室） | 〔部分〕 |
| I5 | <https://weibo.com/ttarticle/x/m/geoshow/id/2309405343730244124688?domain=digital> | 量子位〈把記憶交給 CPU，大模型會變快〉：四個方向各一句、兩組 KV Shrink 測試條件 | 〔部分〕 |
| I6 | <https://blog.csdn.net/csdnnews/article/details/166688978> | 雲棲大會 Intel 專場：「四個組件已在 GitHub 開源」、DSA <64KB、9.66×／1.42×／15× | 〔部分〕 |
| I7 | <https://blog.csdn.net/csdnnews/article/details/166450650> | 陳葆立演講：套件「充分調用 AMX、QAT」 | 〔部分〕 |
| I8 | <https://nb.zol.com.cn/1252/12523108.html> | ZOL：DSA 9.66×、QAT 300GB → 200GB（頁面編碼亂碼） | 〔部分〕 |
| I9 | <https://news.cnyes.com/news/id/6622738> | 鉅亨網：「達 CUDA 方案近 10 倍」「50% 以上壓縮率」 | 〔部分〕 |
| I10 | <https://www.eet-china.com/news/202609262111.html> | 電子工程專輯：四個元件的一句話總述 | 〔摘要〕（讀取逾時，只有搜尋摘錄） |
| I11 | <https://mp.ofweek.com/cloud/a956714896657> | 維科號〈以存代算〉 | 〔標題〕（404） |
| I12 | <https://www.prnasia.com/story/549360-1.shtml> | 憶聯＋Intel KV Cache G3.5（SSD 層），沒有效能數字 | 〔部分〕 |
| I13 | <https://www.intc.com/news-events/press-releases/detail/1783/intel-accelerates-open-infrastructure-for-the-agentic-ai-era> | Intel OCP 新聞稿（2026-10-12）：沒有提到 KV 套件 | 〔部分〕 |
| I14 | <https://semiwiki.com/forum/threads/count-the-work-done-not-just-the-cores-intel.25747/>（原址 <https://newsroom.intel.com/opinion/count-the-work-not-just-the-cores> 已 404） | GPU 只有 50–60% 時間在做有用的工作 | 〔部分〕 |
| I15 | <https://news.pedaily.cn/202610/569846.shtml> | 投資界：「三重考驗」（不是四個難題） | 〔部分〕 |
| I16 | <https://www.intel.com/content/www/us/en/newsroom/news/client-computing/intel-outlines-architectures-for-agentic-ai-at-hot-chips-2026.html> | Hot Chips 2026（Diamond Rapids 等），沒有 KV 卸載 | 〔標題〕 |

### 5.2 論文（對應技術）

| # | 連結 | 內容 | 深度 |
|:--|:--|:--|:--|
| P1 | <https://arxiv.org/abs/2510.10129> | CacheClip（Bin Yang 等；輔助模型在 CPU 上） | 〔摘要〕 |
| P2 | <https://arxiv.org/abs/2502.02789> | SpecPrefill（ICML'25） | 〔摘要〕 |
| P3 | <https://arxiv.org/abs/2405.16444> | CacheBlend（EuroSys'25） | 〔全文，前次；見 `docs/PAPERS_BY_LEVEL.md`〕 |
| P4 | <https://www.usenix.org/system/files/osdi24-lee.pdf> | InfiniGen（OSDI'24） | 〔全文，前次〕 |
| P5 | <https://arxiv.org/abs/2410.21465> | ShadowKV（ICML'25） | 〔全文，前次〕 |
| P6 | <https://arxiv.org/abs/2410.16179> | MagicPIG（ICLR'25） | 〔摘要〕 |
| P7 | <https://arxiv.org/abs/2505.02922> | RetroInfer | 〔摘要〕 |
| P8 | <https://arxiv.org/html/2603.27138v1> | ScoutAttention | 〔摘要〕（只看到搜尋摘錄） |
| P9 | <https://www.alphaxiv.org/zh/abs/2601.12904> | FusionRAG | 〔標題〕 |
| P10 | <https://arxiv.org/abs/2605.01708>；程式碼 <https://github.com/Intelligent-Microsystems-Lab/SplitZip>（`42c7431`） | SplitZip：GPU 上的無損 BF16 KV 壓縮（Triton） | 論文〔摘要〕；README〔全文〕；`codec_gpu.py`〔部分〕 |
| P11 | <https://arxiv.org/abs/2508.19263> | ZipNN 擴充到 FP8／FP4 與 K／V（「up to 62% for BF16 and 83% for FP8」，摘要沒定義是壓縮後大小還是省下的比例） | 〔摘要〕 |
| P12 | <https://arxiv.org/abs/2305.02480> | Intel DSA 量化分析（ASPLOS'24） | 〔摘要〕（只有書目與摘要） |
| P13 | <https://arxiv.org/abs/2511.00739v1>、<https://arxiv.org/abs/2511.00739> | agent 的 CPU 觀點（v1 寫 90.6%；v3 摘要沒有百分比） | 〔摘要〕 |
| P14 | <https://arxiv.org/abs/2605.26297> | Agentic AI Workload Characteristics（IISWC'26） | 〔摘要〕 |

### 5.3 AMD／ROCm 端

| # | 連結 | 內容 | 深度 |
|:--|:--|:--|:--|
| A1 | <https://github.com/ROCm/hipCOMP-core> | hipCOMP：nvCOMP 2.2 的移植，early access | 〔部分〕README |
| A2 | <https://github.com/PAA-NCIC/hipANS> | hipANS：dietGPU 的 HIP 版，有 bf16 float codec | 〔部分〕README |
| A3 | <https://rocm.blogs.amd.com/software-tools-optimization/amd-infinity-context/README.html> | AMD Infinity Context（AIC）（2026-07-22） | 〔部分〕 |
| A4 | <https://blog.lmcache.ai/en/2026/05/12/benchmarking-lmcache-for-multi-turn-agentic-workloads-on-amd-mi300x/> | LMCache 在 2×MI300X 上的 benchmark | 〔部分〕 |
| A5 | <https://rocm.blogs.amd.com/software-tools-optimization/4bit-KV-LMcache/README.html> | 4-bit KV＋LMCache（MI355X，有損） | 〔部分〕 |
| A6 | <https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-10.1-blog/README.html> | ROCm 10.1：hipFile 非同步快速路徑、NUMA 主機記憶體 | 〔部分〕 |
| A7 | <https://github.com/ROCm/rocm-systems/pull/11614> | `hipMemcpy3DBatchAsync` 合成一次 SDMA 送出（未合併） | 〔部分〕 |
| A8 | <https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/api-reference/environment_variables.html> | `HSA_ENABLE_SDMA`、`HSA_ENABLE_PEER_SDMA` | 〔部分〕 |
| A9 | <https://rocmdoc.readthedocs.io/en/latest/Programming_Guides/HIP_Debugging.html> | blit 的延遲 <5 µs、約 80% DMA 頻寬（舊版文件） | 〔摘要〕 |
| A10 | <https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html> | MI300X xGMI：每條 64 GB/s、實測 45–48 GB/s（2025-03-02） | 〔部分〕 |
| A11 | <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf> | MI300X 平台 datasheet（PCIe Gen5 x16 128 GB/s、896 GB/s） | 〔摘要〕（PDF 讀取逾時，只有搜尋摘錄） |
| A12 | <https://github.com/ROCm/mori>、<https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/distributed/vllm-mori-recipe.html> | MoRI／MORI-IO、vLLM 配方 | 〔摘要〕 |
| A13 | <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html>、<https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html> | AITER 與 vLLM V1 的最佳化 | 〔摘要〕 |
| A14 | <https://github.com/ROCm/rocm-systems/tree/develop/projects/hipfile> | hipFile 原始碼 | 〔摘要〕（條件見本專案 D7） |
| A15 | <https://docs.vllm.ai/en/latest/features/kv_offloading_usage.html> | vLLM `OffloadingConnector` 支援 CUDA／ROCm／XPU | 〔部分〕 |
| A16 | <https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/white-papers/epyc-9004-wp-maxlinear-vs-qat.pdf> | EPYC 9004＋MaxLinear 外接壓縮卡 vs QAT | 〔標題〕（只有搜尋摘錄） |
| A17 | <https://www.phoronix.com/news/Linux-6.14-DMA-AMD-AE4DMA> | AE4DMA／PTDMA：為 NTB 設計 | 〔摘要〕 |
| A18 | <https://link.springer.com/chapter/10.1007/978-3-032-30491-9_23> | 〈Characterizing Lossless GPU Data Compression Across AMD CDNA and RDNA Architectures〉 | 〔標題〕（要登入） |

### 5.4 本專案內部（數字的出處）

- `results/m7_write_policy_mi300x/tier_params.json`：CPU、SSD、NFS 的讀寫頻寬
- `docs/phase1_20261008/07_report.md` §7：C6，vLLM CPU 層 4.0–4.1 GiB/s
- `docs/research_20261010_followup/F1_offload_slowdown.md` §4.6、§6：每次 copy 約 3.5 µs，以及建議的修法
- `docs/precision_20261013/08_report_mi300x.md`：FP8／INT4 在各層的還原拆解
- `docs/research_20261010_directions/D7_os.md`：hipFile 的條件，這台不可用
- `docs/MI300X_MLSTEAM.md`：配額，沒有 NVMe
