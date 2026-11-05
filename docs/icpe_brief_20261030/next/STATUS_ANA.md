# STATUS_ANA（icpe-ana，acvlab，分支 `icpe-next-ana`）

> 只由 `icpe-ana` 寫。格式照 NEXT_PLAN §3b。最新的一節在最上面。
> icpe-ana 是分析 session：不跑正式模擬、不動佇列、不改 `code/p3x/*`。
> 本分支只放**沒有 CacheWise** 的東西；CacheWise 上做過的速度與對拍只留在 acvlab。

## 2026-11-06T00:20+08:00 — 補跑計畫（使用者決定：P1–P4 全做、照優先度）；分工；PREREG 偏離 6 由 icpe-data 寫

**使用者 11/6 的決定**（AskUserQuestion＋訊息，本 session）
- 被砍掉的格子**全部補**，照優先度：P1 → P2 → P3 → P4。
- **模擬器的修改由 icpe-ana 做**（Rust 的 W4 雜訊、慢層上限移植；另開 v4，**不動 icpe-sim 正在審的 v3**）；icpe-sim 繼續審查／包裝／G1。
- **PREREG 偏離 6 由 icpe-data 寫**（草稿在下面）；push 由各自的 session 做。

**優先度與時程**〔推算，11/6 00:00；TraceLab 在 Rust 上的速度與記憶體 NOT_MEASURED〕
| 優先 | 內容 | 預計 |
|:--|:--|:--|
| P1 | 截尾版規則（O-DROP／O-NOW／O-ALL 的 -H1800／-H3600；ICPE 稿 §7.3 的 Mooncake 主結果用 H3600）＋**全部格子**都用 7 個 TTL（不只 ≥10% 的格子） | 11/6 下午–11/7 |
| P2 | nvme807（Cake）＋只載入（4 個慢層），從代表格擴到全部格子 | 11/7–11/8 |
| P3 | `wt`＋預取進 GPU，從代表格擴到全部格子 | 11/8–11/9 |
| P4 | W4 雜訊（照 PREREG 第 6 項與 §3 B；範圍＝上限 ≥15% 的格子） | 移植＋對拍約 1 天、跑約 1 天 → 約 11/9–11/10；趕得上 11/16 全文、趕不上 11/9 |

**前提**：Python 正式佇列照舊跑完（SemiAnalysis 約 11/6 04:00、FreeInference 6 週約 04:30、Bailian 約 06:00、TraceLab c4 約 11/6 中午〔推算〕）；Rust 版要等 icpe-sim 在 STATUS_SIM 寫 G1／xcheck PASS 才用於正式結果。

**分工**
| session | 做什麼 |
|:--|:--|
| icpe-sim | 審查 v3、包裝成 `x1_sweep.csv`、G1＋xcheck；之後審查 icpe-ana 的 v4（雜訊、慢層上限） |
| icpe-ana | v4：W4 雜訊（逐位元重現 Python `random.Random`）、慢層上限（`SlowOrder`）＋逐位元對拍；量 TraceLab 的 Rust 時間與記憶體；每個優先度的 job 清單與時間推算 |
| icpe-data（acvlab） | **寫 PREREG 偏離 6**；Python 佇列跑完後用 Rust 跑：TraceLab 的 ≥10% 補跑與 `sens_rep tl_*`、CacheWise 的 P1–P3（只能在 acvlab）、Mooncake 的 P1 |
| icpe-trx（TRX40） | SemiAnalysis 的補跑與 `sens_rep sa_*`；SemiAnalysis、FreeInference、Bailian 的 P1–P3（Rust 在 TRX40 上要另外編、另外對拍） |

**PREREG 偏離 6 草稿**（給 icpe-data；照 §6 的格式補 `date -u` 與「寫的當下看過哪些正式結果」）
1. **換用 Rust 引擎**：`x1 sweep --engine rust`（或 `$X1_ENGINE=rust`；程式在 `code/p3x/x1fast/`＝v3 原樣複製，從 icpe-sim 寫進 STATUS_SIM 的驗證 commit 起；不支援的功能——雜訊、慢層上限、STAIR——該 job 自動改用 Python；icpe-sim 00:1x 的訊息），輸出和 `x1_sim.py` 逐位元相同（G1、xcheck、icpe-ana 的對拍 `results/icpe_next/ana/`）；不改任何格子的結果，所以不改分析計畫。
2. **補回偏離 4 砍掉的格子**（取代偏離 4 第 1、2、4 點的縮小，照優先度執行）：
   - P1：CPU 規則加回 `O-DROP／O-NOW／O-ALL -H1800／-H3600`（截尾版；照 `--dead-h` 的既有定義只報告，不改主判定）；全部格子的 ② 都用 7 個 TTL（0.3／1／3／10／30／100／300 s，TTLA、TTLB）。④ 的 GPU 規則集合是否也擴到 7 個 TTL：**要使用者決定**（偏離 5 的理由：④ 不變才保證「補跑只會讓上限變小」）。
   - P2：nvme807（cake）與只載入（ssd／b32／nfs／nvme807）擴到全部紀錄變體 × 主時間設定以外的全部時間設定。
   - P3：`wt` 與預取進 GPU 擴到全部格子。
   - P4：W4 雜訊照 PREREG 第 6 項與 §3 B（兩種模型、兩種抽法、λ、10 seed；範圍＝④ 對 ② 上限 ≥15% 的每一格、每個元件）。
3. 判準（15%／5%、平均 TTFT 為主指標）不變。

## 2026-11-05T23:58+08:00 — Rust 精確引擎（逐位元相同、快 30× 以上）；請 icpe-sim 審查與驗證

**做完什麼**（細節：[`ana/RUST_NOTES.md`](ana/RUST_NOTES.md)、[`ana/EXPLORATION_LOG.md`](ana/EXPLORATION_LOG.md)、[`ana/LITERATURE.md`](ana/LITERATURE.md)；彙總 CSV：`results/icpe_next/ana/`，每列有 run_id、ts、host）
- 把 `x1_sim.py` 的 `Engine` 改寫成 Rust：`code/icpe/ana_x1fast/`（`v1_port`、`v2_nextreq_cache`、`v3_prefix_fork`）。**沒有改 `code/p3x/` 任何檔**。
- 涵蓋：17 個 CPU 規則（＋PRED、O-DROP-H 截尾版）、GPU 層關閉／wb／wt、GPU 規則 LRU／BEL／TTLA／TTLB、預取到 CPU／GPU、cake／load、多個時間設定。**沒有**：W4 雜訊、慢層容量上限、STAIR。
- **逐位元對拍**（逐請求的 TTFT、還原時間、各層塊數陣列＋`summarize` 的每一欄）〔實測〕：
  - Bailian coder：19 規則 × 9 種 GPU 設定 × CPU 8／64 × cake／load，278／278 相同（v1）；v2、v3 也相同（`results/icpe_next/ana/rust_difftest_jobs.csv`）。
  - Mooncake conversation：68／68 相同（v1、v2；v3 見 CSV）。
  - SemiAnalysis N16 s1 **整個正式 block**：逐一時間設定、照正式 `jobs_for` 切法兩種跑法，各 7,344 列，Rust 的 `mean_ret` 和 `origin/icpe-next-trx` 的 `sa_main/sa_N16_s1` 在其 CSV 精度內**全部相同**（`sa_block_grid.csv`）。
- **速度**〔實測〕：單執行緒中位數 31–35×（Bailian 35.0×、Mooncake 30.9×）；同一行程 16 執行緒再 11.5×；`target-cpu=native` +8%（輸出不變）。PyPy 2.0×、CPython 3.14／Cython／mypyc <1×。
- 近似方法（內插、只看命中率、racing、抽樣 session、兩階段）都會翻 15%／5% 判定或省不到 → 不建議（`approx_eval_sa.csv`；SemiAnalysis 未補跑，探索性）。

**使用者 11/5 的決定**（AskUserQuestion，本 session）：「並行驗證、接手未開始的」——**正在跑的 Python 佇列不中斷**；Rust 版驗證通過後，只用在**還沒開始**的 block 與補回被砍掉的格子（補跑前先登記 PREREG 偏離）。

**需要 icpe-sim 做什麼**（程式歸你；我不改 `code/p3x/`）
1. 審查 `code/icpe/ana_x1fast/v3_prefix_fork/src/main.rs`（v3 含 v2 的快取；共用前段用 `X1FAST_FORK=1` 開）。
2. 包裝：讓 Rust 產出和現在 `x1 sweep` 完全相同欄位的 `x1_sweep.csv`。目前的做法（`code/icpe/ana_bench/difftest.py`）：Python 準備 views／jobs／紀錄的二進位檔 → Rust 跑 → 讀回逐請求陣列填進原本的 `View` → 呼叫原本的 `summarize`。〔推算〕約半天。
3. 驗證：G1（新功能關閉的 4,416 列 × 29 欄逐位元相同）＋xcheck（xc_sa、xc_bl）＋ TraceLab 一個 block 的時間與記憶體（NOT_MEASURED）。過了寫進 STATUS_SIM。
4. 之後若要用在第二批（W4 雜訊要逐位元重現 Python 的 `random.Random`、慢層上限要 `SlowOrder`），再移植。

**需要 icpe-data／icpe-trx 做什麼**：**現在不用動**。icpe-sim 的驗證寫進 STATUS_SIM 之後，用在還沒開始的 block（例如 TraceLab 的 ≥10% 補跑、`sens_rep tl_*`）；正在跑的照舊。

**需要使用者決定什麼**
- 被砍掉的格子要補哪些（截尾版、7 個 TTL 的全面版、nvme807、只載入、wt、預取進 GPU…）；每一項都要先登記 PREREG 偏離。
- 換引擎本身不改結果（逐位元相同），建議在 PREREG 加一句紀錄（從哪個 commit 起改用、用什麼驗證），不算分析計畫的偏離。
