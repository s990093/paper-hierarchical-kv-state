# STATUS_ANA（icpe-ana，acvlab，分支 `icpe-next-ana`）

> 只由 `icpe-ana` 寫。格式照 NEXT_PLAN §3b。最新的一節在最上面。
> icpe-ana 是分析 session：不跑正式模擬、不動佇列、不改 `code/p3x/*`。
> 本分支只放**沒有 CacheWise** 的東西；CacheWise 上做過的速度與對拍只留在 acvlab。

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
