# STATUS_DATA（icpe-data，acvlab）

> 只由 `icpe-data` 寫。格式照 NEXT_PLAN_20261104 §3b：日期時間／做完什麼（commit、run_id）／卡住什麼（完整錯誤）／需要 icpe-sim 做什麼／需要使用者決定什麼。最新的一節在最上面。

## 2026-11-07T02:35+08:00 — acvlab 的 P4：Bailian、CacheWise 的 B2 全部完成；TraceLab B2 跑到 1／6；**B2 倍數：CacheWise 1.01、TraceLab c1 1.10、Bailian traceA 1.58／1.71** → ETA 約 06:15〔推算〕

- **進度**：佇列 pid 2621451（48 workers），沒有 block 失敗。
  - `bailian_traceA_P_B2`、`bailian_traceA_M_B2` 跑完，02:01:41 結束。
  - 18 個 `cachewise_*_B2`：02:01:41 → 02:15:39，全部 rc=0。
  - `tracelab_c1_P_B2`：02:15:39 → 02:30:35，rc=0，run `20261107-021539-icpe-data-p4_b2-tracelab_c1_P_B2`。
  - 現在跑 `tracelab_c1_M_B2`。還剩 5 個 TraceLab B2 block。
- **核對**：這個佇列到目前跑完的 23 個 block 全部通過：`kept_jobs`＝`count`、`unmatched_cells=0`、`py_fallback_jobs=0`、`export RUN_DIR=` 空的、沒有 npz、`exit_code`＝0。
  - 23 個是 4 個 Bailian B2、18 個 CacheWise B2、`tracelab_c1_P_B2`。
  - TraceLab 是 `prep=cache_hit`。
- **B2 對 B1 的倍數**：每個 job 平均 `run_s`，B2 除以 B1，資料來自兩個 run 的 `rust_timing.jsonl`。
  - B1 用 35 workers 跑，B2 用 48 workers，所以倍數裡也含 48 個 worker 互搶的效應。CacheWise 只有 1.01，可見這個效應在 acvlab 上不大。

  | 範圍 | B1 job 數 | B1 每 job run_s | B2 job 數 | B2 每 job run_s | 倍數 |
  |:--|--:|--:|--:|--:|--:|
  | `bailian_traceA_P` | 44,895 | 1.418 | 43,800 | 2.239 | 1.58 |
  | `bailian_traceA_M` | 33,945 | 1.773 | 32,850 | 3.030 | 1.71 |
  | `cachewise_N64*`＋`N72*`（12 個 block） | 175,320 | 0.191 | 170,450 | 0.194 | 1.01 |
  | `tracelab_c1_P` | 8,815 | 4.293 | 8,600 | 4.701 | 1.10 |

  - 以牆鐘時間算：`tracelab_c1_P` 的 B1 是 1,125 s × 35 ÷ 8,815 ＝ 4.47 執行緒秒／job，B2 是 896 s × 48 ÷ 8,600 ＝ 5.00，倍數 1.12。
  - 加上 00:50 那節的 Bailian coder 2.46，B2 變慢的程度因資料集而異。TRX40 的 Mooncake toolagent 約 1.9（STATUS_TRX 02:25，含 46 workers 的 SMT 效應）。
- **ETA**〔推算〕：
  - 剩 5 個 TraceLab B2 block。用各自 B1 的牆鐘速率（c1_M 4.34、c2_P 5.74、c2_M 5.51、c4_P 6.75、c4_M 6.26 執行緒秒／job）乘上倍數，除以 48。
  - 倍數用 1.0／1.12／1.3 → **05:50／06:14／06:50**。
  - 都在 08:00 之前，所以降回 35 workers 的切換不會觸發。佇列結束後那支腳本會印 `SWITCH ABORTED: queue … exited before 08:00` 後退出，這是正常的。
  - 之後 `collect --plans p4_b1 p4_b2`、核對 manifest、commit、push，約 20～30 分鐘。
- **TRX40 的 6 個 Mooncake block**（STATUS_TRX 02:25，31a28798）：完成 4／6。icpe-trx 估 07:10～08:15 進 git〔推算〕。
  - 進 git 之後，我把 6 個目錄 checkout 進 icpe-next-data。
  - 兩邊都齊了之後，通知 icpe-main 開始寫 P4 的結果。

## 2026-11-07T01:00+08:00 — acvlab 的 P4 深夜加到 48 workers（使用者 11/7 00:5x 同意）；08:00 之後在 block 邊界降回 35

- **使用者決定**（本 session）：我提「深夜把 35 個加到約 48 個，約快三分之一〔推算〕」，使用者回「可以」。也要 TRX40 用更多核心：已轉給 icpe-trx 判斷（TRX40 只有 48 個硬體執行緒，36 workers 約用 45 CPU）。
- **切換**（`launch/switch_p4_workers_20261107.sh`，log `runs/switch-p4-w48-20261107.log`）：
  - `p4_b2/bailian_coder_M_B2` 00:52:20 rc=0（run `20261107-002011-icpe-data-p4_b2-bailian_coder_M_B2`）。
  - 下一個 `bailian_thinking_P_B2` 剛開始，00:52:21 就停下舊佇列（pid 1788262），沒有殘留行程。那個 run 寫上 `STOPPED_NOT_USED`。
- **新佇列**：pid **2621451**，`--workers 48`，同樣的 plan 和 glob（不含 6 個 Mooncake block），git HEAD c014e76，log `queue-rust-P4-w48-noMC-20261107-005221.log`。
  - `bailian_thinking_P_B2`、`bailian_thinking_M_B2` rc=0。
  - `bailian_traceA_P_B2`（run `20261107-005246-…`）：`kept_jobs=43800`（＝`count`）、`unmatched_cells=0`、`py_fallback_jobs=0`、`RUN_DIR=` 空的。
  - CPU〔推算〕：48＋12 個摘要行程＋1 ≈ 61。切換前整台 load 約 61，其中我們約 44。
  - 監看：`watch_queues.sh 2621451`。
- **降回 35**：`switch_p4_workers_20261107.sh 2621451 … 35 08:00` 已在等（log `runs/switch-p4-w35-morning-20261107.log`）。08:00 之後第一個 block 邊界重開成 35 workers。
- **ETA**〔推算；假設 48 個執行緒的每執行緒速度不變，08:00 之後用 35 個〕：
  - TraceLab／CacheWise 的 B2 倍數若是 1.0／1.5／2.5 → **06:00／07:50／12:55**。
  - 全程用 35 個的話是 07:56／10:28／15:33。

## 2026-11-07T00:50+08:00 — acvlab 的 P4：B1 全部完成；**B2 在 Bailian 慢約 2.5 倍** → ETA 改為 11/7 08:15～16:00〔推算，範圍大〕

- **進度**：佇列 pid 1788262，沒有 block 失敗。
  - `p4_b1` 交給 acvlab 的 block 全部 rc=0。最後一個是 `tracelab_c4_M_B1`，23:18:02 結束。
  - `p4_b2` 從 23:18 開始。`bailian_coder_P_B2` 00:20:11 rc=0；`bailian_coder_M_B2` 跑到 24,327／33,390。
- **新發現：B2 的 job 比 B1 慢**。用兩個 run 的 `rust_timing.jsonl` 比（`run_s` 是單一執行緒的秒數）：

  | block | job 數 | run_s 中位數 | run_s 合計 |
  |:--|--:|--:|--:|
  | `p4_b1/bailian_coder_P_B1` | 45,633 | 0.268 | 52,047 |
  | `p4_b2/bailian_coder_P_B2` | 44,520 | 0.444 | 130,072 |

  - 每個 job 平均約 2.5 倍。
  - SA 在 TRX40 沒有這個現象：`sa_p4_b1` 的 sweep 合計 10,701 s、`sa_p4_b2` 9,758 s（STATUS_TRX 21:15），job 數相近。
  - 慢多少看紀錄而定，原因還沒查（B2 是每次呼叫都重抽，v5 在 Rust 裡模擬 CPython 的 set 順序）。不影響結果的正確性：`py_fallback_jobs=0`，Rust 對 Python 的 B2 對拍 0 差（STATUS_ANA 43e7ffc）。
- **ETA**〔推算〕：
  - acvlab 還剩 B2 的 Bailian 118,020 個 job（用 2.46 倍）、CacheWise 171,920、TraceLab 103,110。
  - TL 和 CW 的 B2 倍數還沒量。用 1.0／1.5／2.5 倍估：**08:15／10:47／15:52**。
  - CacheWise 的 B2 在 Bailian 之後跑，TraceLab 的 B2 最後跑。量到之後再縮小範圍。
- **TRX40 的 6 個 Mooncake block**：00:45 還沒進 git（icpe-trx 估約 3 h，是用 B1 的速率算的）。其中 4 個是 B2，可能也比較慢。已傳訊息請 icpe-trx 回報進度。

## 2026-11-06T19:50+08:00 — acvlab 的 P4 佇列已重開（不含交給 TRX40 的 6 個 Mooncake block）；ETA 更正為 11/7 約 04:40–05:35〔推算〕

- **切換**（`switch_p4_mooncake_trx_20261106.sh`，log `runs/switch-p4-mooncake-trx-20261106.log`）：
  - 舊佇列跑完 `p4_b1/tracelab_c2_P_B1`：18:49:39 → 19:39:40，rc=0，run `20261106-184939-icpe-data-p4_b1-tracelab_c2_P_B1`。
  - 下一個 `tracelab_c2_M_B1` 19:39:40 剛開始，19:39:43 就停下舊的行程群組（pid 3531487），沒有殘留行程。那個 run（`20261106-193940-…-tracelab_c2_M_B1`）寫上 `STOPPED_NOT_USED`。
- **新佇列**：pid **1788262**，19:39:43 開跑，git HEAD 58fdfae（程式和 f828e79 相同，之後只改了文件）。
  - 指令：`run --plans 'p4_b1:conversation_*,bailian_*,cachewise_*,tracelab_*' 'p4_b2:bailian_*,cachewise_*,tracelab_*' --engine rust --workers 35`。
  - log：`queue-rust-P4-w35-noMC-20261106-193943.log`。
  - 第一個 block `p4_b1/tracelab_c2_M_B1`（run `20261106-193943-icpe-data-p4_b1-tracelab_c2_M_B1`）：`kept_jobs=13826`（＝`count`）、`unmatched_cells=0`、`py_fallback_jobs=0`、`export RUN_DIR=`（空的）、`prep=cache_hit 7.4s lite=True`。
  - 監看：`watch_queues.sh 1788262`，load 監看繼續。
- **ETA 更正**：19:20 那節寫 03:45，偏早。原因是 TraceLab c2 的實測速率比 c1 慢。
  - `tracelab_c2_P_B1`：3,001 s × 35 ÷ 18,286 ＝ **5.74 執行緒秒／job**（c1 是 4.47）。
  - c4 還沒量到。P1 時（Rust、同樣列數），c4 的時間介於 c1 和 c2 之間：`tl_p1` 的 `_ttl` 是 1,752／2,036／1,904 s，`_trunc` 是 1,477／1,865／1,527 s。
  - 所以 c4 用 4.8～5.74 估：acvlab 剩 33 個 block，約 114～125 萬執行緒秒 → **11/7 約 04:40～05:35〔推算〕**。
- TRX40（STATUS_TRX、icpe-trx 訊息）：已 merge 58fdfae（313a9aff），正在跑 Mooncake 的 Python 對 Rust 子集對拍；sa_p4 收完（約 21:00〔推算〕）、重做 G0 之後開跑那 6 個 block。

## 2026-11-06T19:20+08:00 — **P4 的 6 個 Mooncake block 交給 TRX40**（最終清單）；acvlab 的佇列在下一個 block 邊界重開、不跑這 6 個

**依據**
- 使用者 11/6 約 19:10 在 icpe-main 說：「那帮我在 RTX 上面继续跑 P4 的任务，可以直接交给他。」RTX 指 TRX40，「他」指 icpe-next-trx。icpe-main 用訊息轉達，要 icpe-data 決定搬哪些 block。
- icpe-trx 的使用者約 19:15 確認 **TRX40 只接 Mooncake**，不接 Bailian、TraceLab、CacheWise（STATUS_TRX 19:20，afc6e762）。CacheWise 本來就一律不離開 acvlab。
- TRX40 已經有 Mooncake 原始紀錄，sha256 和 STATUS_DATA 相同：conversation `b8cbb061…`、toolagent `48a2db1a…`。所以不用搬任何資料，也不用重建載入快取。

**交給 TRX40 的 block（最終清單）**：acvlab 都還沒開始跑。

| plan | block | job 數 |
|:--|:--|--:|
| p4_b1 | toolagent_P_B1 | 196,718 |
| p4_b1 | toolagent_M_B1 | 148,738 |
| p4_b2 | conversation_P_B2 | 205,160 |
| p4_b2 | conversation_M_B2 | 153,870 |
| p4_b2 | toolagent_P_B2 | 191,920 |
| p4_b2 | toolagent_M_B2 | 143,940 |
| | **合計** | **1,040,346** |

- 在 TRX40 用 `run --plans 'p4_b1:toolagent_*' 'p4_b2:conversation_*,toolagent_*' --engine rust --workers 36`。dry-run 要剛好是上表 6 個 block。
  - `p4_b1` 的 `conversation_*` 已在 acvlab 跑完，**不要**放進 glob。
- **TRX40 的核對**（照 sa_p4）：
  - merge 後重做 G0，要 IDENTICAL；code_sha 要是 `3280b1c8799b+rs62b0932b`。
  - 每個 block 檢查：`kept_jobs`＝`count`、`unmatched_cells=0`、`py_fallback_jobs=0`、`RUN_DIR` 是空的。
  - 開跑前，icpe-trx 會在 TRX40 用 Mooncake toolagent 的 B1＋B2 子集做 Python 對 Rust 的對拍（STATUS_TRX 19:20）。
- **收結果**：icpe-trx 在 TRX40 跑 `collect --plans p4_b1 p4_b2`，push 到 `icpe-next-trx`。
  - TRX40 的完成清單裡只有這 6 個，所以只會產生 `results/icpe_next/b/p4_b1/toolagent_{P,M}_B1/`、`p4_b2/{conversation,toolagent}_{P,M}_B2/` 這 6 個目錄，`host` 欄是 TRX40 的 hostname。
  - 和 acvlab 的目錄不重疊。icpe-data 再把這 6 個目錄 checkout 進 `icpe-next-data`，讓 P4 全部在同一個分支。
- **TRX40 跑不完時**：在 STATUS_TRX 寫清楚哪些 block 沒做，acvlab 再用同樣的 glob 補跑。同一個 block 不會兩台都跑。

**acvlab 怎麼跳過這 6 個**
- 正在跑的佇列（pid 3531487）在開跑時就載入了完成清單和 block 清單，所以要重開。
  - `launch/switch_p4_mooncake_trx_20261106.sh`：等目前的 `p4_b1/tracelab_c2_P_B1` 跑完、下一個 block 剛開始時，停下行程群組。
  - 剛開始的那個 run 寫上 `STOPPED_NOT_USED`。
  - 然後重開：`run --plans 'p4_b1:conversation_*,bailian_*,cachewise_*,tracelab_*' 'p4_b2:bailian_*,cachewise_*,tracelab_*' --engine rust --workers 35`，log 是 `queue-rust-P4-w35-noMC-<ts>.log`。新的 pid 之後補在這裡。
- **dry-run 核對**（19:16，在 tl_c2_P_B1 還沒跑完時做；沒有建立 run 目錄）：
  - 已完成跳過 28 個，要跑 34 個（含正在跑的 tl_c2_P_B1；重開時它已完成，會被跳過）。
  - 28＋34＋6＝68，等於 p4_b1＋p4_b2 的全部 block，合計 2,297,276 個 job。
  - 要跑的 34 個裡沒有任何 toolagent，也沒有 conversation 的 B2。34 個都是 `RUN_DIR=`（空的）、都有 `--jobs-from`。
- **時間**〔推算，用 18:40 的速率〕：
  - acvlab 剩 34 個 block、528,146 個 job、約 113 萬執行緒秒，35 個執行緒 → **11/7 約 03:45**（原本 06:50）。
  - TRX40：104 萬個 job × 0.374（acvlab 的 Mooncake 速率）÷ 36 ≈ 3.0 h。TRX40 跑 Mooncake 的速度 NOT_MEASURED。

## 2026-11-06T18:40+08:00 — acvlab P4 的 ETA 更新：11/7 約 06:50〔推算〕（15:40 估 03:00 偏低）

- `p4_b1/tracelab_c1_P_B1` 跑完：18:17:08 → 18:35:53，rc=0，8,815 個 job。`unmatched_cells=0`，`py_fallback_jobs=0`。
- **實測速率**（block 時間 × 35 ÷ job 數，用這個佇列和舊佇列 `conversation_P_B1` 的 log）：

  | 紀錄 | 執行緒秒／job | 用了多少 job |
  |:--|--:|--:|
  | Mooncake conversation | 0.374 | 369,288 |
  | Bailian | 1.324 | 167,184 |
  | CacheWise | 0.264 | 176,832 |
  | TraceLab（只有 c1） | 4.467 | 8,815 |

- 剩下的 job × 上面的速率（toolagent 用 conversation 的速率；B2 用 B1 的速率；TL c2／c4 用 c1 的速率）：約 154 萬執行緒秒，35 個執行緒約 12.3 h → **11/7 約 06:50 跑完〔推算〕**。TraceLab 約占剩餘的 58%。
  - 15:40 估 03:00，偏低的原因：Mooncake 用 0.31（實測 0.374），Bailian 沒有單獨估。
- 執行緒數不變，維持約 44 個 CPU 的上限。

## 2026-11-06T18:20+08:00 — TraceLab 的 P4 用上 icpe-sim 的載入快取；PREREG 同步偏離 7 更正與偏離 8

- **TL 快取**：第一個 TraceLab 的 P4 block `p4_b1/tracelab_c1_P_B1`（run `20261106-181708-icpe-data-p4_b1-tracelab_c1_P_B1`）的 stdout 印 `prep=cache_hit 10.0s lite=True`。
  - 用的是 icpe-sim 90dae78 建的快取（STATUS_SIM 15:50）。不需要 icpe-sim 再做什麼。
- **acvlab P4 進度**（18:17）：這個佇列（pid 3531487）25 個 block rc=0，加上舊佇列的 `conversation_P_B1`，共 713,304／2,297,276 個 job（31%）。沒有 rc≠0。
  - 已完成：Mooncake conversation（P、M）、Bailian 全部、CacheWise 全部的 B1。接著是 TraceLab 6 個 B1 block，然後 toolagent，再全部的 B2。
  - TraceLab 的雜訊每個 job 要多久還沒量到；`tracelab_c1_P_B1` 跑完後再更新 ETA（15:40 的估計是 11/7 約 03:00〔推算〕）。
- **PREREG**：從 `origin/icpe-next-a` 6b84a5c checkout 檔案，內容和 `origin/icpe-next-main` 相同（274 行）。
  - 和上一版相比只有新增：19 行新增、0 行刪除。新增的是「偏離 7 更正」和「偏離 8」，兩段都是 icpe-main 寫的。

## 2026-11-06T15:40+08:00 — acvlab 的 P4 用不存 npz 的程式重開；TRX40 的 sa_p4 15:16 開跑

- 舊佇列跑完 `p4_b1/conversation_P_B1`：15:03:42 → 15:35:06，rc=0，210,289 個 job，1,884 s。這個 block 有存 npz，約 10 GB，留著。
  - 15:35:07 停在 `conversation_M_B1` 剛開始的地方，那個 run 留了 `STOPPED_NOT_USED`。
- **新佇列**：pid 3531487，`run --plans p4_b1 p4_b2 --engine rust --workers 35`，git HEAD f828e79，log `queue-rust-P4-w35-nonpz-20261106-153507.log`。
  - 第一個 block `conversation_M_B1`：cmd 是 `export RUN_DIR=`，req/ 是空的。
  - `kept_jobs=158999`，和 `count` 相同；`unmatched_cells=0`、`py_fallback_jobs=0`。
- 監看：`watch_queues.sh 3531487`，以及 load 監看。
- **時間**〔推算〕：Mooncake 的雜訊 job 實測約 0.31 執行緒秒（210,289 個 job × 35 ÷ 1,884 s）。
  - TraceLab 用 P1 的 3.9 s 加約 25%（雜訊的額外成本）來估，約佔總量 2／3。
  - 合計約 150 萬執行緒秒，35 個執行緒約 12 h → **11/7 約 03:00** 跑完。順序是全部 B1 先跑，再跑 B2。
- **TRX40**（STATUS_TRX 15:20，cdea0d4）：merge f828e79 之後，15:16:47 開跑 `sa_p4_b1 sa_p4_b2`，36 個執行緒。
  - 48 個 block 都是 `RUN_DIR=` 空的。第一個 block 的 kept 等於 `count`。
  - 他們估約 19:45 跑完。

## 2026-11-06T15:25+08:00 — P4 不存逐請求 npz（icpe-trx 的請求；兩台都適用）→ **icpe-trx 請 merge 本節的 commit 後再開始 sa_p4**

- **問題**（STATUS_TRX 15:15，8d9d34b；他們的數字）：`next_b_queue` 設 `RUN_DIR=<run>/req`，x1_sim 每個 job 都會存一個 npz。
  - SA 的 P4 有 1,434,342 個 view，約 1.2 TB〔推算〕；TRX40 的 /ssd4 只剩 548 GiB。
  - acvlab：/ssd6 剩 2.0 TB（共用）。第一個 P4 block（Mooncake conversation）跑到一半就存了 5.5 GB、96,850 個檔。TL 每個 job 約 4 MB 以上，P4 的 TL 約 20 萬個 job → 約 0.9 TB〔推算〕。
- **P4 需不需要 npz**：PREREG（§3 B、偏離 6／8）的 P4 只要求 seed 間的中位數、最小–最大、平均±標準差，`next_sim_analyze.py` 的雜訊分析只讀 sweep CSV。
  - 全文沒有要求 bootstrap 或逐請求的資料（`grep bootstrap／x1_boot／逐請求` 只找到偏離 6 的對拍說明）。
  - npz 只有 `x1_boot.py`（成對 bootstrap）會用到。之後如果要某幾格的逐請求資料，可以用同樣的參數重跑那幾格：程式是確定性的，結果會逐位元相同。
- **改法**：`next_b_queue.py` 在 P4 的 block 加 `no_npz=True`，`RUN_DIR` 設成空字串，x1_sim 就不存 npz（`RUN_DIR` 在 x1_sim／x1_rust／p3_fvar 裡只用在這裡）。
  - 其他 plan 不變，空跑核對過。
  - 不改 icpe-sim 的程式。CSV、run_id 都照舊。
- **acvlab 的 P4 佇列**（pid 3462686，15:03 開始）用的是開跑時載入的舊程式，會存 npz。
  - 處理：`restart_p4_nonpz_20261106.sh` 等目前的 `p4_b1/conversation_P_B1` 跑完、下一個剛開始時停下，用新程式重開同樣的 plan。
  - 已經存的那份（Mooncake，約 10 GB）留著。
- **icpe-trx**：請 merge 本節的 commit，用 `--dry-run` 確認 `export RUN_DIR=`（空的），再開始 `sa_p4_b1 sa_p4_b2`。

## 2026-11-06T15:10+08:00 — merge v5＋`--jobs-from`（d3a6376）、G0 PASS → **icpe-trx 可以開始 sa_p4**；acvlab 的 P4 開跑

- STATUS_SIM 14:25：v5（B2 進 Rust，程式 4a4baa4）G1 PASS。STATUS_SIM 15:05：`--jobs-from`（程式 1bf35bd，疊在 v5 上）G1 PASS。
- **先試合併**：在 detached worktree 合併 icpe-next-data HEAD 和 d3a6376，模擬程式和 1bf35bd 沒有差異。
  - **G0（Rust）PASS**：run `20261106-150227-icpe-data-g0-rust-merged-1bf35bd`，`--workers 32`，**128,064 格 0 個不同**，sweep 24.2 s。執行檔是 `src_sha256=62b0932b…`，code_sha 是 `3280b1c8799b+rs62b0932b`。
  - 正式 merge：**47707e0**，tree `4920721…` 和試合併相同。
  - merge 後 `count --plans p4_b1 p4_b2 sa_p4_b1 sa_p4_b2` 仍是 3,301,642。
- 共用 venv 是 Python 3.12.15，所以 B2 交給 Rust（v5 只在 3.12 時才用 Rust）。
- **icpe-trx 可以開始 sa_p4**：merge `origin/icpe-next-data`（含 47707e0 或之後的 commit）→ 在 TRX40 重新編譯並做 G0、xc_sa 等對拍 → `run --plans sa_p4_b1 sa_p4_b2 --engine rust --workers 36`。
  - `count` 應該是 48 個 block、1,004,366 個 job。
  - 每個 block 開跑後，看 stdout 的 `jobs_from=`（kept／dropped／unmatched）：kept 要等於 `count` 的數字。
- **acvlab**：`run --plans p4_b1 p4_b2 --engine rust --workers 35`（68 個 block、2,297,276 個 job），先跑完全部 B1 再跑 B2。pid 和 log 開跑後補。
  - TL 快取：icpe-sim 用 d3a6376 重建 tl_c1／c2／c4，15:01 開始，約 21 分鐘。TL 的 block 在佇列後段，那時已經建好。

## 2026-11-06T14:15+08:00 — **acvlab 的 P1–P3（偏離 6）全部完成**；TRX40 的 SA P1–P3 也完成

- TraceLab 的 P3 `*_wt` 已收進 git，3 個 block，各 7,344 列，程式是 363024a（wt 修正）：

  | block | run | sweep 時間 |
  |:--|:--|--:|
  | `tl_c1_wt` | `20261106-131021-icpe-data-tl_p3-tl_c1_wt` | 2,083 s（含自己建準備 1,187 s） |
  | `tl_c2_wt` | `20261106-134505-icpe-data-tl_p3-tl_c2_wt` | 880 s |
  | `tl_c4_wt` | `20261106-135947-icpe-data-tl_p3-tl_c4_wt` | 819 s |

- **acvlab 的 P1–P3 都在 git 裡**：
  - `{tl,cw,old,bl}_p1`：36 個 block。
  - `{tl,cw,old,bl}_p2`：36 個 block。
  - `{tl,cw,old,bl}_p3`：36 個 block。
  - `sens_rep` 的 `tl_cacheprefix`、`tl_t128k`。
- **TRX40**（STATUS_TRX 14:20，c8e554f）：
  - `sa_p1`、`sa_p2`、`sa_p3` 各 24 個 block。其中 `sa_p3 *_wt` 是用 776486c 重跑的，12 個 block、88,128 列（c3e5681）。
  - `sens_rep sa_t128k` 也完成了。
- **P4**：icpe-trx 在 TRX40 跑 `sa_p4_b1`、`sa_p4_b2`，用 36 個執行緒（使用者在那邊的選擇）。
  - v5 和 `--jobs-from` 都 merge、G0 PASS 之後，我會在這裡寫「icpe-trx 可以開始 sa_p4」和 commit。
- acvlab 目前沒有佇列在跑，等 icpe-sim 的 v5（約 15:00）和 `--jobs-from`（約 16:15）。

## 2026-11-06T14:05+08:00 — P4 的排程寫進 `next_b_queue.py`（plan `p4_b1`／`p4_b2`／`sa_p4_b1`／`sa_p4_b2`）；共 3,301,642 個 job

- 把 icpe-main 的選格清單同步進本分支：`results/icpe_next/main/p4/`，4 個檔，取自 `origin/icpe-next-main` c868ef2，內容沒有改。
- `next_b_queue.py`：
  - `p4_blocks()` 依 PREREG 偏離 8 的清單，每個紀錄 × 模型（P／M）× 抽法（B1／B2）排一個 block。
    - 每個（格, 元件）展開成 ④ 要的 (CPU 規則, GPU 規則) 組合，整理成 job 清單 `(trace, rule, cpu_GiB, policy, gpu_policy, slow, gpu)`。
    - 開跑時把清單寫進 `<run>/jobs.csv`，x1 sweep 加 `--jobs-from <run>/jobs.csv`。這個選項要等 icpe-sim 實作好、我 merge 並做完 G0 之後才有。
    - 雜訊參數：
      - P 的 B1：λ 0、0.1、0.2、0.3、0.4；P 的 B2：λ 0.1–0.4。
      - M 的 B1：λ 0–0.3；M 的 B2：λ 0.1–0.3。
      - seed 都是 0–9。λ＝0 只在 B1 的 block 跑一次（自檢）。
  - `count` 支援雜訊參數和 job 清單的過濾，規則照 icpe-sim 說的：job 的某一個 view 在清單裡就保留。
- **`count` 的結果**（`count --plans p4_b1 p4_b2 sa_p4_b1 sa_p4_b2`）：
  - `p4_b1` 1,164,816、`p4_b2` 1,132,460、`sa_p4_b1` 509,256、`sa_p4_b2` 495,110，**合計 3,301,642**。
  - 比 13:55 我自己數的 3,214,596 多 2.7%。原因是 sweep 會把 GDSF 按 (慢層, GPU 閒／忙) 拆成好幾個 job，我自己數的時候當成一個。以 `count` 為準。
- **分工**：`p4_b1`、`p4_b2` 在 acvlab 跑；`sa_p4_b1`、`sa_p4_b2` 給 TRX40。

## 2026-11-06T13:55+08:00 — P4（W4 雜訊）的規模；使用者決定照 PREREG 全跑；請 icpe-sim 加 `--jobs-from`、先做 v5

**輸入**：icpe-main 的選格清單（`origin/icpe-next-main` c868ef2，`results/icpe_next/main/p4/p4_cells_distinct.csv`；PREREG 偏離 8）。975 格、2,582 個（格, 元件）。
- 我看過的內容：這個檔的表頭和前 4 列，其中有 4 個 Bailian 格的 `max_headroom` 值；以及我自己數 job 用的欄位（trace、rule、cpu、slow、gpu、components）。沒有看其他上限表。

**job 數**（`scratchpad/p4count.py`；直接用 `x1_sim` 的 `COMPONENT_POLICIES`／`NON_ORACLE`／`POLICIES`／`noise_specs`）
- 每個元件要跑的組合，照 `next_sim_analyze.py` 的 ④ 定義：
  - O-ORD／O-DROP／O-PF：該元件的 CPU 規則 × 15 個 GPU 不預測規則（LRU＋TTLA／TTLB × 7 個 TTL）。
  - O-GPU：10 個 NON_ORACLE × GPU BEL。
  - O-ALL(report)：(O-ALL, BEL)。
- 雜訊設定 142 種：
  - P：λ 0、0.1、0.2、0.3、0.4；M：λ 0、0.1、0.2、0.3。
  - λ＞0 的每一個 × B1、B2 × 10 個 seed；λ＝0 各 1 個。
  - 142 種裡 B2 佔 70 種。
- 不重複的 job：22,638 × 142 ＝ **3,214,596**。
  - 依紀錄：Mooncake 1,387,340、SemiAnalysis 967,446、CacheWise 336,540、Bailian 322,482、TraceLab 200,788。
- 用現有的乘積參數排 block 會多跑：依 (trace, rule, cpu, slow) 分組要 509 個 block、5,253,006 個 job（多 63%）；依 (trace, rule) 分組要 10,521,916 個。
- 〔推算〕時間：用 P1 `_ttl` 實測的 Rust 每 job 平均秒數。
  - 各紀錄：TraceLab 3.92 s、Mooncake 0.246、Bailian 0.208、CacheWise 0.117、FI 0.046；SemiAnalysis 用 TRX40 `sa_p2` 的中位數 0.62。
  - 合計約 183 萬執行緒秒。SemiAnalysis 在 TRX40 跑，acvlab 約 10 h、TRX40 約 5 h。

**使用者決定**（AskUserQuestion，11/6 13:5x）：**照 PREREG 全跑**，不縮小 GPU 規則，也不登記新偏離。

**需要 icpe-sim 做什麼**（已傳訊息）
1. v5（B2 進 Rust）優先審查、G1、push。沒有 v5，160 萬個 B2 job 會退回 Python，跑不完。
2. `x1 sweep --jobs-from <csv>`：只保留清單裡的 job（`jobs_for` 展開後過濾）。
   - 清單欄位 trace, rule, cpu_GiB, policy, gpu_policy, slow, gpu。單一 view 的 job 要比對 (slow, gpu)。
   - 沒給這個選項時，行為不變；給了要驗證：保留下來的 job，結果和不過濾時逐格相同。

**我的排法**（等上面兩項）
- 一個紀錄變體 × 雜訊模型（P／M，λ 清單不同）× 抽法（B1／B2）＝一個 block。
- 先跑 B1（現有程式已能用 Rust），B2 等 v5。
- SemiAnalysis 交給 TRX40。

## 2026-11-06T13:12+08:00 — merge wt 修正（776486c）、G0（Rust）PASS；TL 的 P3 `*_wt` 開跑

- STATUS_SIM 13:15：icpe-sim 的 wt 修正（選項 A，程式 7193a2d）G1 PASS。
- **先試合併**：在 detached worktree 把 icpe-next-data HEAD 和 776486c 合併。模擬程式和 7193a2d 沒有差異，改動只有 `x1_sim.py`、`main.rs` 各 4 行。
  - **G0（Rust）PASS**：run `20261106-130931-icpe-data-g0-rust-merged-7193a2d`，`--workers 32`，**128,064 格 0 個不同**，sweep 32.5 s。執行檔重新編譯，`src_sha256=e9497446…`。
  - 正式 merge 是 363024a，tree `be27bef…` 和試合併相同。暫時的 worktree 已移除。
- **TL 的 wt**：佇列 pid 3034018，跑 `tl_p3:*_wt`（3 個 block），35 個執行緒，13:10:21 開始，log `queue-rust-TLwt-w35-20261106-131021.log`。監看用 `watch_queues.sh 3034018`。
  - x1_sim.py 改了，準備快取的鍵也變了。icpe-sim 正在用 776486c 重新預建 5 個 TL 快取；預建完成前開始的 block 會自己建（lite 約 5 min）。
- 已經跑完的 CW／old／BL `*_wt` 不重跑：那幾種紀錄沒有 0 塊的模式（STATUS_DATA 10:10）。icpe-sim 的修改前後對照也確認，修正只影響 SA 的 ARC、S3FIFO（相對差 ≤3.6e-5）和計數欄。

## 2026-11-06T13:10+08:00 — acvlab 的 P3 除了 TL wt 都完成；兩個 TL 代表格完成；目前沒有佇列在跑

- 佇列 A3（pid 2650007）：20 個 block 全部 rc=0，11:56:18 → 13:03:23。
- `collect` 35 個 block，已進 git：
  - P3 的 33 個：CW／old／BL 的 `*_wt` 15 個；全部紀錄的 `*_pfgpu` 18 個。
  - `sens_rep/tl_cacheprefix`、`sens_rep/tl_t128k`：各 1,632 列。STATUS_DATA 08:20 更正的漏排，這兩個補完了。
  - 全部 host acvlab、Rust，單一 xz 檔最大 1.30 MB。
- **偏離 6 在 acvlab 還沒做的，只剩 `tl_p3 *_wt` 3 個 block**：等 icpe-sim 的 wt 修正（7193a2d）push、寫進 STATUS_SIM，我再先試合併、做 G0，然後跑。
- **目前 acvlab 上沒有我們的佇列**，CPU 是空的。
- P4 要等 icpe-main 從版本 B（`origin/icpe-next-main` baa508d）選出格子，加上 B2 的 Rust 版通過驗證。

## 2026-11-06T12:55+08:00 — **acvlab 的 P2 完成**（36 個 block，已進 git ed37d48）；B 換成 B'；只剩 A3

- 佇列 B 跑完 `tl_p2/tl_c4_load`（12:28:48）。`switch_B_20261106.sh` 照計畫把它停在 `old_p2/mooncake_load` 剛開始的地方，那個 run 留了 `STOPPED_NOT_USED`。
  - 改開 B'：pid 2811634，log `queue-rust-Bp-w17-20261106-122849.log`。3 個 block 都是 rc=0，12:51:19 結束。
  - 注意：switch 程序印的 pid（2650003）是錯的。B' 的 pid 是 2811634，用 `ps` 核對過。
- **P2**：`collect` 36 個 block → `results/icpe_next/b/{tl,cw,old,bl}_p2/`，共 17.5 MB，單一 xz 檔最大 1.74 MB。
  - 每個紀錄變體：`_nvme807` 2,448 列，`_load` 9,792 列，和 TRX40 的 `sa_p2` 相同。
  - 全部 host acvlab、Rust。
- 現在只剩 **A3**（pid 2650007）：P3 的 pfgpu，以及兩個 TL 代表格。監看用 `watch_queues.sh 2650007`。
- **還沒跑的**：`tl_p3 *_wt`（3 個 block），等 icpe-sim 的 wt 修正（7193a2d）push 和 STATUS_SIM。

## 2026-11-06T12:00+08:00 — CW／old／BL 的 P3 `*_wt` 完成；空出來的執行緒交給 A3（P3 pfgpu＋兩個 TL 代表格）

- **佇列 A2**：15 個 `*_wt` block 全部 rc=0，11:26:47 → 11:55:28。CW 12 個、Mooncake、FI、Bailian 各 1 個。
- **空檔怎麼用**：TL 的 wt 要等修正（約 12:30 以後），A2 的 18 個執行緒空出來。所以：
  - **佇列 A3**：pid 2650007，11:56 開始。跑 `tl_p3:*_pfgpu cw_p3:*_pfgpu old_p3:*_pfgpu bl_p3:*_pfgpu sens_rep:tl_cacheprefix,tl_t128k`，共 20 個 block，18 個執行緒。
    - 第一個 block `tl_c1_pfgpu` 是 `prep=cache_hit 14.4s`。
  - **佇列 B**：照原樣跑完手上的 `tl_p2/tl_c4_load`。由 `/ssd6/hungwei/icpe-next/launch/switch_B_20261106.sh` 在它剛開始 `old_p2/mooncake_load` 時停下，改開 B'（`old_p2:*_load bl_p2:*_load`，3 個 block，17 個執行緒）。
    - 為什麼要換：B 只在開跑時讀一次完成清單，不換的話它會重跑 A3 做過的 block。
  - 空跑核對過：A3 的 20 個 ＋ B' 的 3 個，正好等於 B 除了 `tl_c4_load` 以外還沒做的 23 個，沒有重疊。
- CPU：18＋17 個執行緒，和之前一樣，約 44。
- 監看：`watch_queues.sh 1752970 2650007`。

## 2026-11-06T11:30+08:00 — 佇列 A 照計畫停在 TL wt 之前，改跑 CW／old／BL 的 wt；wt 修正（使用者選 A）驗證中

- 佇列 A 的 P2（`*_nvme807`，18 個 block）在 11:26:44 跑完。
  - 自動程序 v2 在 `tl_p3/tl_c1_wt` 剛開始時把它停下來（11:26:47）。那個 run 留了 `STOPPED_NOT_USED`。
  - 接著開佇列 A2：pid 2477497，`cw_p3:*_wt old_p3:*_wt bl_p3:*_wt`，18 個執行緒，log `queue-rust-A2-w18-20261106-112647.log`。
- 佇列 B 照跑，現在在 `tl_p2/tl_c2_load`。
- 監看換成 `/ssd6/hungwei/icpe-next/launch/watch_queues.sh 1752970 2477497`：每分鐘記錄一次；MemAvailable < 60 GiB 就停；任一個佇列結束就通知。
- **wt 修正**（icpe-sim 的訊息，11/6 11:1x）：
  - 使用者選 A：0 塊的請求在回合結束時，把 session 從 CPU 順序表拿掉。兩種回合結束（只有 CPU、wt）都改，兩個引擎都改。
  - 本地 commit 7193a2d。驗證 run `20261106-111012-icpe-sim-g1-7193a2d`，預計 12:30 完成，過了才 push。
  - 修正後我 merge → G0 → 補跑 `tl_p3:*_wt`（3 個 block）。
  - `x1_sim.py` 改了，準備快取的鍵也會變。icpe-sim 會照同樣的規則重新預建 5 個 TL 快取。

## 2026-11-06T10:50+08:00 — **acvlab 的 P1 完成**（36 個 block，已進 git）；兩個佇列進入 P2

- P1 在 acvlab 的最後一個 block 10:41 結束。`collect` 36 個 block → `results/icpe_next/b/{tl,cw,old,bl}_p1/`，共 15.6 MB，單一 xz 檔最大 1.31 MB。
- 每個紀錄變體的列數：
  - `_ttl`：6,480 列＝8 個新 TTL 的 GPU 規則 × 15 個 CPU 規則 × CPU 9 個大小 × 3 個慢層 × GPU 閒／忙。
  - `_trunc`：5,184 列。
- 筆數：TraceLab 3 個變體、CacheWise 12、Mooncake 4、FI 6、Bailian 3。全部 host acvlab、`--engine rust`。
- 程式版本：
  - `tl_c4_*` 起的 block 用 8af643c（準備快取版）。
  - 之前的 block 用 e3c7d80／acef68c（模擬程式＝1304345）。
  - 兩版逐位元相同（G0、G1）。
- SemiAnalysis 的 `sa_p1` 在 `origin/icpe-next-trx`，24 個 block 已完成（STATUS_TRX 09:45）。
- **所以 5 類紀錄的 P1 都齊了**，icpe-main 可以算 P1 版的上限，然後選 P4 的格子。
- 現在在跑：佇列 A 在 `tl_p2/tl_c1_nvme807`，佇列 B 在 `tl_p2/tl_c1_load`。

## 2026-11-06T10:10+08:00 — Mooncake／FI 也沒有這種模式 → `old_p3 *_wt` 照跑；目前 git 裡的結果都不受 wt 漏洞影響

- icpe-sim 掃描的結果（`20261106-094552-icpe-sim-wtpanic/zero_block_scan_nonnb.csv`；他們的數字）：
  - Mooncake conversation／toolagent × R1／R2：0 次，根本沒有 0 塊的請求。
  - FI 6 週：0 次。有 0 塊的請求，但都是該 session 的第一個請求，不會留下過期的一筆。
- 結論：已在 git 的 wt 結果（`sens_rep` 的 cw／bl／mc／fi `*_wt`），以及 GPU 0 的 H200 t1／p01（也就是 G0 參考，都是 Mooncake），**都不受影響**。
- 佇列 A 的自動程序換成 v2：`/ssd6/hungwei/icpe-next/launch/hold_wt_v2_20261106.sh`。
  - 走到 `tl_p3/tl_c1_wt` 就停，改跑 `cw_p3:*_wt old_p3:*_wt bl_p3:*_wt`（15 個 block）。
  - 只有 `tl_p3 *_wt`（3 個 block）等 icpe-sim 修正，以及使用者對語意的決定。
- `sens_rep tl_wt` 從來沒跑過：11/5 TraceLab 在等記憶體修正。之後改由修正後的 `tl_p3 *_wt` 涵蓋。

## 2026-11-06T10:00+08:00 — wt 的問題是模型漏洞（Python 也會出錯）；TraceLab／舊紀錄的 P3 `*_wt` 先不跑

**icpe-sim 的初步結論**（訊息 11/6 09:5x；run `20261106-094552-icpe-sim-wtpanic`；他們的數字，完整內容會寫進 STATUS_SIM）
- Python 引擎跑同一個 job 也出錯（IndexError，同一行），所以**不是只有 Rust 有這個問題**。
- 原因：一個 session 先有 block，下一個請求變成 0 塊時，`_turn_end_wt` 只把它從 `cpu_set` 拿掉，沒有從 LRU 順序表拿掉，留下過期的一筆。
  - 會預取的規則（O-PF、LRU+RPF）碰到時會直接 panic。
  - 用自己順序表的規則（LRU 家族、O-DROP、O-NOW、ARC、S3FIFO）**不會報錯，但會默默影響** `n_demote`、`cpu_residence_s_mean`；ARC／S3FIFO 的淘汰順序也可能被改變。
- 只影響 wt 和 GPU 0。wb＋GPU（主結果、P1、P2、pfgpu）不受影響。
- 有這種模式的紀錄：SemiAnalysis 每個變體 198 次、TraceLab 34 次（c1_t128k 1 次）；CacheWise、Bailian 0 次；Mooncake／FI 還在查。
- 要修就得決定模型的語意，icpe-sim 會問使用者。修完之後，受影響格子的數字會變。

**我的處理**（只改排程順序，不改程式、不繞過）
- 佇列 A：`/ssd6/hungwei/icpe-next/launch/hold_wt_20261106.sh`。
  - 一走到 `tl_p3/tl_c1_wt` 就停，改跑 `cw_p3:*_wt bl_p3:*_wt`（18 執行緒，13 個 block）。這兩類紀錄沒有這種模式，修不修結果都一樣。
  - `tl_p3 *_wt`、`old_p3 *_wt` 等修正與使用者決定之後再跑。Mooncake／FI 的模式數如果是 0，old 的部分先排進去。
- 佇列 B 照跑：P3 是 pfgpu，屬於 wb。
- **可能受影響、要另外處理的**（已提醒 icpe-sim）：
  - 11/5 用 Python 跑的 `sens_rep` `mc_wt`、`fi_wt`，要看 Mooncake／FI 的模式數。
  - GPU 0 的舊結果，例如 H200 t1／p01，也就是 G0 的參考檔。

## 2026-11-06T09:50+08:00 — TraceLab 準備快取生效；TRX40 的 `wt` block 在 Rust panic（acvlab 的 P3 `*_wt` 也可能碰到）

**準備快取**（icpe-sim 預建 5 個：run `20261106-093010-icpe-sim-prewarm-8b6520f`，09:30–09:42）
- `tl_p1/tl_c4_trunc`（09:39 開始）：`prep=cache_hit 7.7s lite=True`。
- `tl_p1/tl_c4_ttl`（09:35 開始，當時快取還沒好）：`prep=built 274.4s lite=True`。
- 之後的 TraceLab block，準備時間從約 16 min 降到約 10 s。

**TRX40 的失敗**（STATUS_TRX 09:45，origin/icpe-next-trx @ 98b0d2f；他們的數字）
- `sa_p1`、`sa_p2` 各 24 個 block 已完成、已進 git。
- `sa_p3/sa_N16_s1_wt` 在 Rust 的 `src/main.rs:667` panic，佇列照規則停下來。
  - 位置：`demote()` 開預取時 `next_access(bl[bl.len()-1])`，`bl` 是空的。
- 已通知 icpe-sim，請它確認兩件事：
  1. Python 引擎跑同一個 job 會不會也出錯。
  2. 已經跑完、沒有 panic 的格子不會被默默算錯。我的推論：G1／xcheck 的 bl／mc／fi × wt 都是 0 個不同，而且碰到這個情況會直接 panic、不會繼續算。
- **對 acvlab 的影響**：佇列 A 的最後一段是 P3 的 `*_wt`，〔推算〕約 11:00 之後才會開始。
  - 如果在那之前沒有修好、又碰到 panic，佇列 A 會照規則停下來（CLAUDE.md 規則 2），我回報，不繞過。
  - 修好的話，走 G1 → 寫進 STATUS_SIM → 我 merge → G0。
- `sens_rep` 在 `origin/icpe-next-trx` 上目前沒有 `sa_t128k`（STATUS_DATA 08:20 的更正）。

## 2026-11-06T09:35+08:00 — merge 準備快取（8b6520f）、G0（Rust）PASS

- **先試再合**：佇列在跑，每個新 block 開跑時會讀 data worktree 的程式，所以先在暫時的 detached worktree 試合併，跑 G0，通過後才 merge 進 data worktree。
  - 試合併：icpe-next-data HEAD＋8b6520f。和 c7ef740 比，模擬程式沒有差異。
  - sha256 前 7 碼和 icpe-sim 給的對照相同：`x1_sim.py` 2723419、`x1_nb.py` 8251e25、`x1_rust.py` 28f82fd。
  - **G0（Rust）PASS**：run `20261106-093120-icpe-data-g0-rust-merged-c7ef740`，`--workers 8`，**128,064 格 0 個不同**，sweep 114 s。
  - 正式 merge：8af643c。tree `4c38203…` 和試合併的 tree 逐位元相同。暫時的 worktree 已移除。
- **之後開始的 block 都用快取版**：正在跑的 `tl_c2_ttl`、`tl_c2_trunc` 不受影響。
  - icpe-sim 從 09:30 起預先建 5 個 TraceLab 變體的快取：run `20261106-093010-icpe-sim-prewarm-8b6520f`，同時最多 3 個行程。
  - 確認方式：下一個 TraceLab block 的 stdout 看到 `prep=cache_hit … lite=True` 就是命中。
  - 如果看到 `built_saved`，表示快取還沒建好，結果一樣正確，只是那一次照舊要約 16 分鐘。

## 2026-11-06T09:05+08:00 — 使用者決定：全合成的工作負載（ShareGPT、W7 地圖）照計畫，ICPE 之後再做

- 使用者問「人工數據」那塊做了沒有。我的回答分三種：
  1. **合成到達**：對話內部全用真實資料，只合成對話的開始時間。
     - 包括 CacheWise、SemiAnalysis 的固定並行 N，以及 TraceLab c2／c4 的壓縮 2×／4×。
     - 已在主結果裡，P1–P3 正在跑。
  2. **ShareGPT 全合成**（NEXT_PLAN §5.6 的選跑）：沒做。
  3. **W7 參數掃描地圖**（NEXT_PLAN「ICPE 之後」）：沒做。
- **使用者決定**（AskUserQuestion，11/6 09:0x）：「照計畫，ICPE 之後再做」。
  - 11/9 前專心做 P1–P4 與分析。
  - 論文只用合成到達的結果，限制一節寫明「沒有做全合成」。

## 2026-11-06T08:40+08:00 — **主結果全部完成**（acvlab 4 類＋TRX40 的 SemiAnalysis）；佇列拆成兩個；icpe-main 開了

**做完什麼**
- acvlab 最後 3 個主結果 block 用 Rust 跑完，已收進 git（e8bcdaf）：

  | block | run | 列數 | sweep 時間 |
  |:--|:--|--:|--:|
  | `tl_main/tl_c4` | `20261106-075035-icpe-data-tl_main-tl_c4` | 7,344 | 1,950 s（其中約 16 min 是準備） |
  | `old_main/fi6w` | `20261106-082307-icpe-data-old_main-fi6w` | 44,064 | 438 s |
  | `bl_main/bailian3` | `20261106-083025-icpe-data-bl_main-bailian3` | 22,032 | 116 s |

  - 程式都是 e3c7d80（模擬程式＝icpe-sim 1304345），35 個執行緒，host acvlab。
- **主結果 5 類全部齊了**：
  - acvlab：`cw_main` 12、`tl_main` 3、`old_main` 2、`bl_main` 1。
  - TRX40：`sa_main` 12（`origin/icpe-next-trx`）。
- **佇列拆成兩個**（08:25 的 (a)）：交接程序 08:32:25 執行。
  - 舊佇列剛開始的 `tl_p1/tl_c1_ttl`（run `20261106-083222-…`）停掉，留了 `STOPPED_NOT_USED`。
  - 佇列 A：pid 1751802，18 個執行緒，log `queue-rust-A-w18-20261106-083225.log`。
  - 佇列 B：pid 1752970，17 個執行緒，log `queue-rust-B-w17-20261106-083225.log`。
  - 監看：
    - 每分鐘把兩個佇列的進度、記憶體、load 寫到 `mem-guard-rust2-20261106.log`；MemAvailable < 60 GiB 就停兩個佇列。
    - load 連續 10 分鐘 > 140 時通知。
- **icpe-main**（使用者決定：主結果的分析另開 session）：08:33 開。
  - tmux `icpe-main`，Opus 5.5、effort xhigh、remote control。
  - worktree `/ssd6/hungwei/icpe-next/main-ana`，branch `icpe-next-main`，從 e8bcdaf 開出來。
  - prompt：`/ssd6/hungwei/icpe-next/main_prompt_20261106.md`。
  - 範圍：主結果 3 個 TTL 的版本 → P1 版（先核對列數）→ P4 選格 → P2／P3 版。分析的選擇先問使用者，不跑模擬、不動佇列。
  - 它會寫 `STATUS_MAIN.md`。

**接下來**
- icpe-sim 的 TraceLab 準備快取：G1 PASS、push 之後，我 merge、做 G0，再確認新 block 的 log 是 `prep=cache_hit`。
- 每一級跑完就收進 git。

## 2026-11-06T08:25+08:00 — 使用者決定加速 (a)(b)(c)；CPU 上限維持 44

**量到的**：Rust 佇列跑 `tl_c4`。
- 父行程準備 TraceLab：07:50:35 → 約 08:07，約 16 分鐘。這段時間只用 1 個 CPU，另外 35 個執行緒閒著。
- 開跑後：08:11 時 846／6,174（206 s）。
- 〔推算〕P1–P3 還有 18 個 TraceLab block，準備時間合計約 4.8 h，約佔 acvlab 剩下時間的 40%。

**使用者決定**（AskUserQuestion，11/6 08:1x）：三項都做；都不改任何格子或規則，只影響時間。
- **(a) 拆成兩個佇列**：在主結果和 P1 的交接點換。
  - 由 `/ssd6/hungwei/icpe-next/launch/split_queue_20261106.sh` 自動做：
    - 等 log 出現 `start tl_p1/tl_c1_ttl`。
    - 確認 tl_c4、fi6w、bailian3 都 rc=0。
    - 停掉舊佇列，在那個剛開始的 block 留 `STOPPED_NOT_USED`。
    - 改開兩個佇列。
  - 佇列 A（18 執行緒）：每一級的 `*_ttl`、`*_nvme807`、`*_wt`，TraceLab 先。
  - 佇列 B（17 執行緒）：每一級的 `*_trunc`、`*_load`、`*_pfgpu`，CacheWise 先，讓兩邊的準備時間錯開；最後是 `sens_rep:tl_cacheprefix,tl_t128k`。
  - 兩邊都照 P1 → P2 → P3。空跑核對：A 54 個 block、B 56 個，沒有重疊，合起來等於原本的全部 110 個。
  - CPU：18＋17 個執行緒、4＋4 個摘要行程、2 個父行程，約 45 個。
  - 兩個佇列同時用 `X1_RUST_WORK` 是安全的：紀錄檔用內容 sha 命名，同時搬移時有處理；每個 job 的目錄用 mkdtemp。
  - 〔推算〕最多省約 2.4 h：一邊在準備時，另一邊在算。
- **(b) icpe-sim 快取 TraceLab 的準備結果**：已傳訊息請它做。
  - 要求：逐位元相同、G1 PASS。我 merge、做 G0 之後，從下一個 block 開始用；正在跑的不停。
- **(c) W4 雜訊 B2 進 Rust**：已傳訊息請 icpe-ana 寫、icpe-sim 審查包裝。
  - 要在 P4 開跑前完成；沒做好的話，B2 照舊退回 Python。
- **CPU 上限**：使用者說「我怕用太多，會卡到其他人」。**維持約 44**，不提高。

## 2026-11-06T08:20+08:00 — 更正：截斷 128K、快取命中前綴的代表格漏排了（TL 兩個、SA 一個）

**錯在哪**（icpe-data 的疏失；icpe-trx 在 STATUS_TRX 08:06 指出 `sa_t128k` 這一格）
- 偏離 6 寫明：偏離 4 第 4 點的「快取命中前綴」和「截斷 128K」**維持代表格**，不在 P1–P3 裡。
- 這些代表格是 `sens_rep` 的 block。
  - acvlab 上已完成 `cw_cacheprefix`、`cw_t128k`、`fi_t128k`（11/5）。
  - **`tl_cacheprefix`、`tl_t128k`、`sa_t128k` 還沒跑。**
- 我 07:55 排 Rust 佇列時沒有排 `sens_rep`；08:10 給 icpe-trx 的步驟寫了「`sens_rep sa_*` 不跑」。
  - 理由「P2／P3 已涵蓋全部格子」只對 nvme807、只載入、wt、預取進 GPU 成立，**對 `t128k`、`cacheprefix` 不成立**。

**改正**
- `next_b_queue.py run --plans` 接受 `plan:glob1,glob2`，表示這個 plan 只跑名稱符合的 block。例：`sens_rep:tl_cacheprefix,tl_t128k`。
  - `--only` 仍然對全部 plan 生效。
  - 空跑檢查過。
- **acvlab**：不中斷正在跑的佇列（中斷會浪費正在跑的 block）。
  - 佇列（pid 1629288）乾淨跑完，也就是 log 有 `end bl_p3/bailian3_pfgpu rc=0`、沒有 BLOCK FAILED 時，自動接著跑：
  - `run --plans sens_rep:tl_cacheprefix,tl_t128k --engine rust --workers 35`
  - 兩個都是代表格（TraceLab c1 的變體），預計各 10–30 分鐘〔推算〕。
- **給 icpe-trx**：08:10 第 4 步的指令改成：
  - `next_b_queue.py run --plans sa_p1 sens_rep:sa_t128k sa_p2 sa_p3 --engine rust --workers 32`
  - 需要先 merge 本節的 commit，才有 `plan:glob`。
  - 如果佇列已經照舊指令開跑，就不要中斷；跑完再補 `run --plans sens_rep --only sa_t128k --engine rust --workers 32`。
  - `sens_rep` 其餘 4 個 `sa_*`（nvme807、load、wt、pfgpu）仍然不跑：`sa_p2`／`sa_p3` 已涵蓋。

## 2026-11-06T08:10+08:00 — 使用者決定：SemiAnalysis 的 P1–P3 由 TRX40 裝 Rust 自己跑；主結果分析交給新的 session

**使用者決定**（AskUserQuestion，11/6 08:0x，在 icpe-data）
- 問題：SemiAnalysis 的 P1–P3（349,704 個 job）在哪裡跑。
- 背景：STATUS_TRX 07:55 寫明，TRX40 用 Python 約要 80 h，11/9 前跑不完；TRX40 沒有 Rust。
- 選項有三個：搬到 acvlab 的 Rust 佇列；TRX40 裝 Rust 自己跑；TRX40 照舊計畫用 Python。
- **使用者選「TRX40 裝 Rust 自己跑」。**
  - 這也回答了 STATUS_TRX 07:55 的「還沒決定」：改用 Rust 跑偏離 6 的 P1–P3，舊計畫的 `sa_main_ttl7`、`sens_rep sa_*` **不跑**。
  - 不跑舊計畫的理由：P2／P3 已涵蓋全部格子，包括代表格。

**另一個使用者決定（同一時段）**
- 主結果齊了以後，另開一個 session（Opus、effort xhigh、tmux、remote control）專門做主結果的分析。
- icpe-data 繼續跑實驗。新 session 的名稱和範圍，開的時候寫在這裡。

**給 `icpe-trx`（TRX40）的步驟**：照順序做；任何一步出現不同的格子或失敗，就停下來、記完整錯誤、寫進 STATUS_TRX。
1. **merge**：`git merge --no-ff origin/icpe-next-data`，要含本節的 commit。
   - 裡面有 icpe-sim 的 47a4617（Rust 引擎＋W4 雜訊 B1，G1 PASS）和 `next_b_queue.py run --engine rust`。
   - 模擬程式應和 icpe-sim 的 1304345 逐位元相同；用 `git diff 1304345 HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` 確認是空的。
2. **裝 Rust**：用使用者權限（rustup，不用 sudo），裝在 TRX40 自己的目錄。
   - 版本和 acvlab 相同：**rustc 1.99.1 stable**（cargo 1.99.1），`--profile minimal` 就夠。
   - `code/p3x/x1fast` 沒有外部套件（Cargo.lock 只有自己），可以 `--offline` 編譯。
   - 環境變數：
     - `X1_CARGO`、`CARGO_HOME`、`RUSTUP_HOME` 指到 TRX40 的工具鏈。程式的預設值是 acvlab 的路徑，TRX40 上不存在。
     - `X1_RUST_WORK` 指到 TRX40 自己的目錄。
   - 第一次 `--engine rust` 會自己編譯。
3. **對拍**（三項都要 0 個不同）：
   - (a) **G0（rust）**：照 acvlab run `20261106-074940-icpe-data-g0-rust-merged-1304345` 的 `cmd.sh`，換成 TRX40 的路徑，用 `--engine rust`。
     - 和 `results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz` 用 `next_b_g0.py` 比。
     - 要 `G0_RESULT IDENTICAL`，128,064 格。
   - (b) **xc_sa**：用 `--engine rust` 重跑，和你們 Python 版的 xc_sa 逐格比。
   - (c) **一個完整的正式 block**：照 `results/icpe_next/b/sa_main/sa_N16_s1/manifest.json` 的 `cmd` 加 `--engine rust`，放新的 run 目錄。
     - **不要**用 `next_b_queue.py run`：它會跳過已完成的 block。
     - 和同一個 manifest 的 `x1_sweep.csv.xz` 用 `next_b_g0.py` 比；它可讀 .xz，也會去掉 host 欄。要 7,344 列 0 個不同。
     - 理由：Rust 在 TRX40 上是新編的執行檔，結果和 CPU 型號無關只是〔推論〕（STATUS_SIM 03:15），要用整個 block 實測。
4. **跑**：`next_b_queue.py run --plans sa_p1 sa_p2 sa_p3 --engine rust --workers 32`。
   - TRX40 的環境變數照舊（`NEXT_B_TAG=icpe-trx` 等）。
   - CPU：約 1.25×workers，32 時約 41 個，和原本 Python 的 40 worker 差不多。workers 用 4 的倍數（STATUS_SIM 04:32）。
   - 記憶體保護和監看照 `sa_main` 的做法。
   - job 數〔算術〕：P1 135,648、P2 124,416、P3 89,640。和 `next_b_queue.py count --plans sa_p1 sa_p2 sa_p3` 核對。
5. **收**：每個 plan 跑完就 `collect`，commit 並 push 到 `icpe-next-trx`，在 STATUS_TRX 寫 run_id、時間、記憶體。
   - 不解讀上限：分析由新的分析 session 做。
   - Rust 在 TRX40 上的每 job 時間記下來（`rust_timing.jsonl`），這是 NOT_MEASURED 的項目。

## 2026-11-06T07:55+08:00 — 使用者決定「兩個都停，全改 Rust」；merge 47a4617、G0 PASS；Rust 佇列開跑

**使用者決定**（AskUserQuestion，03:25 提問、07:4x 回答）：「兩個都停，全改 Rust」。
- 停掉兩個 Python 佇列，剩下的 block 全部用一個 Rust 佇列跑，CPU 總量約 44。
- 已登記：**PREREG 偏離 6 補記**（813617a，`date -u` 2026-11-05T23:49:08Z）。改的是偏離 6 第 1 點「只用在還沒開始的 block」。
  - 同一個 block 不會混用兩種引擎。Python 跑到一半的輸出不收、不用。
- 回答時的進度和提問時差很多：tl_c4 從 454 變成 3,835／6,174，fi6w 從 22,693 變成 34,993／37,044。
  - 照現況重算〔推算〕：tl_c4 用 Python 還要約 3 h，用 Rust 重跑約 30 min。改用 Rust 仍然比較快，所以照做。

**做完什麼**
- **停掉兩個 Python 佇列**：07:49:20 對兩個 process group 送 `kill -TERM`，2 秒內全部結束。
  - `tl_main/tl_c4`（run `20261106-022134-…-tl_c4`）停在 3,847／6,174。
  - `old_main/fi6w`（run `20261105-222021-…-fi6w`）停在 35,022／37,044。
  - 兩個 run 目錄都留了 `STOPPED_NOT_USED` 說明檔，沒有寫進完成清單。
- **merge `origin/icpe-next-sim` @ 47a4617**（e3c7d80）：Rust 引擎加 W4 雜訊 B1。驗證見 STATUS_SIM 06:14。
  - merge 後的模擬程式和 1304345 逐位元相同。
- **G0（Rust）PASS**：run `20261106-074940-icpe-data-g0-rust-merged-1304345`，`--workers 32`。
  - **128,064 格，0 個不同**；`py_fallback_jobs=0`；sweep 29.7 s。
  - code_sha `a27559138492+rs43dbe828`。

**現在在跑什麼**：Rust 佇列，pid 1629288，07:50:35 開始。
- 指令：`next_b_queue.py run --plans tl_main old_main bl_main tl_p1 cw_p1 old_p1 bl_p1 tl_p2 cw_p2 old_p2 bl_p2 tl_p3 cw_p3 old_p3 bl_p3 --engine rust --workers 35`
- log：`/ssd6/hungwei/icpe-next/runs/queue-rust-w35-20261106-0750*.log`
- 順序：
  1. 先補完主結果：tl_c4 → fi6w → bailian3。
  2. 再照 P1 → P2 → P3。每一級裡依 TraceLab → CacheWise → Mooncake／FI → Bailian。
- CPU：35 個 Rust 執行緒、8 個摘要行程（35//4）、1 個父行程，約 44 個。
  - 摘要行程數 ÷ 執行緒數＝0.23。icpe-sim 量到 TraceLab 需要約 0.17（STATUS_SIM 04:32），所以夠用。
- 監看：
  - 每分鐘記 MemAvailable、load、行程樹 PSS 和進度，寫到 `mem-guard-rust-20261106.log`。MemAvailable < 60 GiB 就停佇列。
  - load 連續 10 分鐘 > 140 時通知。
- 還沒排的：
  - P4（W4 雜訊）：要先用 P1 之後的上限選 ≥15% 的格子。
  - SemiAnalysis 的 P1–P3：在 TRX40。plan `sa_p1`／`sa_p2`／`sa_p3` 已在 `next_b_queue.py`，icpe-trx 要先 merge 47a4617、在 TRX40 編譯並對拍 Rust。

## 2026-11-06T03:25+08:00 — tl_c2 完成；merge Rust 引擎（875fb2b）、G0（Rust）PASS；佇列工具加 `--engine rust`

**做完什麼**
- **`tl_main/tl_c2`**：run `20261105-172713-icpe-data-tl_main-tl_c2`，exit 0，6,174 個 job，sweep 32,060 s（40 worker）。
  - 已收進 `results/icpe_next/b/tl_main/tl_c2/`：7,344 列（386f84d）。
- **merge `origin/icpe-next-sim` @ 875fb2b**（304fd72）：照 STATUS_SIM 2026-11-06T03:15（Rust G1、自檢、14 個 block 的 xcheck 都 PASS）。
  - merge 後 `code/p3x`、`p3_fvar.py`、`p3_platform.py` 和 0b0ef1e 逐位元相同（`git diff` 是空的）。
- **G0（Rust 引擎）PASS**：run `20261106-031718-icpe-data-g0-rust-merged-0b0ef1e`。
  - 設定：`--engine rust --workers 3`，新功能全關，共用 venv，nice 10。
  - 結果：4,416 列 × 29 欄＝**128,064 格，0 個不同**（`G0_RESULT IDENTICAL`）。`py_fallback_jobs=0`，sweep 296 s。
  - stdout 記了執行檔：`bin_sha256=001f462d…`、原始碼 `src_sha256=c867782a…`，code_sha 欄是 `8c1f52f7dd1f+rsc867782a`。
  - Python 預設引擎這次沒有另做 G0：程式和 0b0ef1e 逐位元相同，而 icpe-sim 已在同一台、同一個共用 venv 對 0b0ef1e 跑過 Python 的 G1（4,416 × 29，0 個不同）。
- **`next_b_queue.py run --engine {py,rust}`**（預設 py，行為不變）：
  - rust 時 x1 sweep 加 `--engine rust`，逐 job 時間記在 `<run>/rust_timing.jsonl`。
  - context.txt 多記一行 `engine:`。
  - 空跑 `tl_p1 --only tl_c1_ttl --engine rust --workers 32` 檢查過，`check_p1` 通過。

**現在在跑什麼**（都是 Python，merge 前開跑的 block 不受影響：worker 用 fork，程式在開跑時已載入）
- 40 worker：`tl_main/tl_c4`，02:21 開始，03:16 時 454／6,174。
- 4 worker：`old_main/fi6w`，03:16 時 22,693／37,044。

**需要使用者決定什麼**：Rust 要不要接手已經開始的 `tl_c4`。→ 已決定，見 07:55 那節。

## 2026-11-06T00:45+08:00 — 偏離 6 的 P1–P3 排程已加；job 數和 icpe-ana 的推算完全相同

- **`next_b_queue.py`** 新增：
  - plan `{cw,tl,old,sa,bl}_p1`：每個主結果 block 拆成兩個。
    - `_ttl`：8 個新 TTL 規則 × 15 個 CPU 規則（`NON_ORACLE`＋`COMPONENT_POLICIES`）。
    - `_trunc`：6 個截尾規則 × 16 個 GPU 規則。
  - plan `_p2`：每個主結果 block 拆成 `_nvme807`、`_load`。
  - plan `_p3`：每個主結果 block 拆成 `_wt`、`_pfgpu`。
  - P2／P3 的規則集合同 `sens_rep`。
  - `count` 子指令：直接呼叫 `x1_sim.jobs_for` 數 job，不讀紀錄、不跑模擬。
  - `check_p1`：`count` 和 `run`（plan 名稱以 `_p1` 結尾時）會先核對三件事，不符就停：
    - 補跑的 CPU 規則正好是 `x1_sim.NON_ORACLE` ∪ `COMPONENT_POLICIES`。
    - `RIVALS` 包含在 `NON_ORACLE` 裡。
    - 截尾規則都在 `POLICIES` 裡。
  - 起因：icpe-sim 11/6 提醒，④ 缺任何一個 TTL 時 `next_sim_analyze.py` 會靜默整列跳過。
- **job 數**（run `20261106-004257-icpe-data-count-p1p3`，stdout 在 run 目錄）：
  - 主結果每個變體 6,174，和實際 sweep 印的 `jobs=` 相同，這個數法可信。
  - 每個變體多出的 job：P1 `_ttl` 6,480、`_trunc` 4,824；P2 `_nvme807` 2,214、`_load` 8,154；P3 `_wt` 6,174、`_pfgpu` 1,296。
  - 40 個變體合計：P1 452,160＋P2 414,720＋P3 298,800＝**1,165,680**。
  - **和 icpe-ana 的推算（STATUS_ANA 00:45）每一項都相同。**
- **出表後的檢查**（P1 分析時做）：每份紀錄 × 每個元件的列數，要等於只有 3 個 TTL 時的列數。少了就停下來，不收進結果。
- **使用者決定（11/6 01:00）**：換成 Rust 後，CPU 總用量維持約 44。
  - icpe-sim 的預告：`--workers N` 實際用約 1.25×N 個 CPU（N 個 Rust 執行緒＋N/4 個摘要行程）。
  - 所以 Rust 的佇列設 N＝32 左右（約 40 個 CPU）。還有其他佇列同時跑時，合計也不超過約 44。
- 還沒做的：
  - 加 `--engine rust`。要等 icpe-sim 的包裝 commit 和 STATUS_SIM 的 G1／xcheck PASS。
  - 分析時排除重複的格子：P1 和 `cw_main_ttl7` 重疊，P2／P3 和 `sens_rep` 的代表格重疊。分析 P1 時用主結果＋`_p1`，不放 `_ttl7`。

## 2026-11-06T00:15+08:00 — PREREG 偏離 6 已寫（補回砍掉的格子 P1–P4＋Rust 引擎）；使用者選「④ 也擴 7 個 TTL」

**做完什麼**
- **PREREG 偏離 6**（`date -u` 2026-11-05T16:09:01Z）：照 STATUS_ANA 00:20 的草稿（`origin/icpe-next-ana` @ 1ea44c9；Rust 引擎的寫法用 icpe-sim 的包裝 `x1 sweep --engine rust`）。補上三樣：`date -u`、「寫的當下看過哪些正式結果」、「icpe-data 的解讀」。
- **使用者在 icpe-data 的決定**（11/6 00:10 左右，AskUserQuestion）：P1 全部格子用 7 個 TTL 時，**④ 的 GPU 規則也擴到 7 個 TTL**（兩邊同一組）。
  - 主結果的分析**不加** `--rival-only-gpu`。偏離 5 的算法只並列報告。
  - 我在提問時向使用者說明了：偏離 5 寫的「方向保守」只就選格規則而言；② 比 ④ 多規則可挑時上限會偏低，對我們的主張反而有利。
- 我寫的「解讀」（使用者可以改）：
  - P2／P3 的規則集合同 `sens_rep`（17 個 CPU 規則 × 8 個 GPU 規則）。
  - P4 用 P1 之後的上限選 ≥15% 的格子。
  - 某個優先度沒做完時，報告值照偏離 4／5，並標明版本。

**需要 h200 做什麼**：偏離 6 在本分支這個 commit。要不要同步進 `icpe-next-a` 由你決定（偏離 5 你是用 2e5757d 同步的）。

**需要 icpe-sim 做什麼**：照 STATUS_ANA 的分工審查、包裝 Rust 引擎，並在 STATUS_SIM 寫 G1／xcheck PASS。主結果的分析用原本的「兩邊同一組 GPU 規則」，不需要新選項。

**接下來我做什麼**：Python 佇列照舊跑完。Rust 驗證過、我 merge 並 G0 通過後，用 Rust 跑以下幾項，順序照 P1 → P4：
- TraceLab 的 P1（取代原本的 ≥10% 補跑）和 `sens_rep tl_*`。
- CacheWise 的 P1–P3（只能在 acvlab）。
- Mooncake 的 P1。
- 佇列工具會加 P1 的 plan：每個主結果 block 補 8 個 TTL 規則 × 15 個 CPU 規則，加上 6 個截尾規則 × 16 個 GPU 規則。

## 2026-11-06T00:00+08:00 — Mooncake 主格完成；FI 6 週在跑；收到 icpe-ana 的 Rust 引擎通知

**做完什麼**
- **`old_main/mooncake`**：run `20261105-085744-icpe-data-old_main-mooncake`，exit 0，24,696 個 job，sweep 48,151 s（4 worker，git HEAD f4f5668）。
  - 已收進 `results/icpe_next/b/old_main/mooncake/`：29,376 列，conversation／toolagent × R1／R2 各 7,344 列。

**現在在跑什麼**
- 40 worker：`tl_main/tl_c2`，4,262／6,174 個 job（23:53）。worker 的 PSS 合計 34 GiB，沒有累積。之後 `tl_c4`。
- 4 worker：`old_main/fi6w`（22:20 開始，git HEAD 63c072f，含 maxtasks），8,748／37,044 個 job（5,089 s）。
  - 照目前速度〔推算〕還要約 4.6 h → 11/6 約 04:30。之後 `bl_main/bailian3`。

**icpe-ana 的 Rust 引擎**（STATUS_ANA 23:58，origin/icpe-next-ana aee1527）：收到，**現在不動**。
- 正在跑的 Python 佇列照舊。
- 要等 icpe-sim 審查、包裝、過 G1／xcheck／TraceLab 測試並寫進 STATUS_SIM 之後，我才 merge → G0，而且只用在還沒開始的 block。
- 被砍掉的格子要補哪些，由使用者決定；補跑前先登記 PREREG 偏離。

## 2026-11-05T21:10+08:00 — load 超過 100，使用者決定不降 worker；新開分析 session icpe-ana

- 20:22 起 acvlab 的 load 在 115–118（96 核）。上升的是其他使用者新開的工作，我們維持 44 個 worker，都用 `nice 10`。
  - **使用者決定（11/5 21:00）：不降。** load 監看門檻改成連續 10 分鐘 >140 才通知。
  - 記憶體正常：可用約 300 GiB，TraceLab worker 合計 37–41 GiB。
- **新 session `icpe-ana`**（使用者要求；tmux `icpe-ana`，Opus、high effort、remote control）。
  - 獨立的分析 session，只讀、只分析，不跑正式模擬、不動佇列、不碰 `/ssd6/hungwei/icpe-next/data`。
  - worktree `/ssd6/hungwei/icpe-next/ana`，branch `icpe-next-ana`（只在本機，從 a60f05e 開出；push 前會先問使用者）。
  - 開場指示在 `/ssd6/hungwei/icpe-next/ana_prompt_20261105.md`。

## 2026-11-05T18:20+08:00 — TraceLab c1 完成（maxtasks=4 後記憶體不再累積）；c2 在跑

**做完什麼**
- **`tl_main/tl_c1`**：run `20261105-115504-icpe-data-tl_main-tl_c1`，exit 0，6,174 個 job，sweep 19,670 s（40 worker，程式 2dc6c8a）。
  - 已收進 `results/icpe_next/b/tl_main/tl_c1/`（7,344 列，xz）。
- **記憶體**（`mem-guard-tl-20261105.log`）：worker 的 PSS 合計在 c1 和 c2 前 358 個 job 都在約 39–56 GiB 上下來回，**沒有隨 job 數增加**。maxtasks=4 有效，不改成 2。MemAvailable 約 290–315 GiB。

**現在在跑什麼**
- 40 worker：`tl_main/tl_c2`（17:27 開始）。之後 `tl_c4`。
- 4 worker：`old_main/mooncake`，10,467／24,696 個 job（9.2 h）。worker 都在約 99% CPU，慢是 job 本身重，不是被搶。

**時程**〔推算，用 tl_c1 每個（CPU 容量, 規則）的 job 耗時當尺度〕
- `tl_c2`：還要約 6–7 h（同一組的 job 是 c1 的 1.28–1.42 倍）→ 約 11/6 01:00。`tl_c4`（npz 是 c1 的 1.5 倍）再約 9–10 h → 約 11/6 中午。
- Mooncake 剩下的組，跟 TL 的比例差很多（四分位 0.02–0.20）→ 還要 2.5–23 h。之後 `fi6w`、`bailian3` 也在這條 4 worker 的佇列上。
- 打算：40 worker 那條在 TL 主格＋TL 的 ≥10% 補跑＋`sens_rep tl_*` 做完後，若 4 worker 那條還有沒開始的 block，就在 block 交界把它停掉，剩下的改給 40 worker 跑（兩條佇列不同時排同一個 block）。

## 2026-11-05T11:56+08:00 — 敏感度格 19 個完成；merge 54fa2d8（maxtasks）、G0 PASS；TraceLab 重開

**做完什麼**
- **`sens_rep` 的 CacheWise／Bailian／Mooncake／FI 代表格**：19 個 block 全部 exit 0（10:05 → 10:58，40 worker），已收進 `results/icpe_next/b/sens_rep/`（commit 039a8cf）。
  - 內容：nvme807、只載入、`wt`、預取進 GPU；CacheWise 另有快取命中前綴；CacheWise 和 FI 另有截斷 128K。
  - 程式：7539f01（merge 54fa2d8 之前）。
- **merge** `origin/icpe-next-sim` @ 984dc4f → **2dc6c8a**。
  - 程式＝54fa2d8：`x1 sweep --maxtasks`，預設 4，worker 每 4 個 job 重新 fork 一次。
  - 依據：G1 PASS，run `20261105-100635-icpe-sim-g1-54fa2d8`（STATUS_SIM 11:45）。
- **G0（merge 後）PASS**：run `20261105-114540-icpe-data-g0-merged-54fa2d8`（40 worker，stdout 印 `maxtasksperchild=4`，sweep 541 s）：**128,064 格，0 個不同**。
- **TraceLab 重開**：`run --plans tl_main --workers 40`（c1 從頭跑），log `queue-tl_main-w40-k4-20261105-115504.log`。
  - 記憶體監看：`mem-guard-tl-20261105.log`，每 2 分鐘記一次 worker 的 PSS 合計和 job 數。MemAvailable <60 GiB 就自動停掉這條佇列。
  - 前幾百個 job 的 PSS 合計若還在漲，就照 icpe-sim 的建議改 `X1_MAXTASKS=2`（k 不影響結果）。
- **4 worker 的佇列**：現在跑的 Mooncake block 是 merge 前開跑的（7539f01），Mooncake 本來就沒有累積。接下來的 `fi6w`、`bailian3` 開跑時會讀到 54fa2d8。兩版輸出逐位元相同：G1 和本節的 G0 都驗過。
- icpe-trx 在 TRX40 上也對 54fa2d8 做了 G0 和 xc_sa，都是 0 格不同（STATUS_TRX 9410199）。它 merge 2dc6c8a 後開 `sa_main`（40 worker）。

## 2026-11-05T10:10+08:00 — **TraceLab 停了**：worker 記憶體每個 job 累積（同 icpe-trx 10:05）；先跑非 TL／SA 的敏感度格；等 icpe-sim 的 maxtasksperchild

**發生什麼**（acvlab 量到的）
- TraceLab c1（run `20261105-084552-icpe-data-tl_main-tl_c1`，40 worker）：
  - worker 的 PSS 合計：開跑時 20.4 GiB → 827 個 job 後 **108.1 GiB**（每個 worker 1.38–6.14 GiB），**約每個 job 0.1 GiB**。
  - MemAvailable 從 352 降到 176–253 GiB（其他使用者的用量也在變）。
- 推算〔算術〕：照這個速度，一個 TraceLab block 的 6,174 個 job 要約 **670 GiB**。合計大致跟 job 數走、和 worker 數無關，所以**不管開幾個 worker 都裝不下 440 GiB**，而且 swap 已滿。
  - 這個 block 注定跑不完，sweep 也不能中途接續。
  - 為了不把整台共用機器的記憶體吃光，**10:04 停在 827／6,174**，run 目錄留了 `ABORTED`，沒有 collect。
- 同時 4 worker 的 Mooncake（不是 nb 紀錄）：3,896 個 job 後每個 worker 仍是 0.53–0.59 GiB，**沒有明顯累積**，所以繼續跑。

**現在 acvlab 在跑什麼**
- 40 worker：`sens_rep --only 'cw_*' 'mc_*' 'fi_*' 'bl_*'`（19 個小 block，每個只有 2 個 CPU 容量），log `queue-sens-small-w40-20261105-100505.log`。
- 4 worker：`old_main`（Mooncake → FI 6 週）→ `bl_main`。
- 記憶體保護：MemAvailable <60 GiB 時，自動停掉我的兩條佇列（`mem-guard-20261105.log`）。

**需要 icpe-sim 做什麼**：同 icpe-trx 的請求（STATUS_TRX 00cb757 10:05）——`x1_sim.py` 的 Pool 加 `maxtasksperchild`，或找出跨 job 留下的東西。
- acvlab 的數字供參考：TraceLab 約 0.1 GiB／job，SemiAnalysis（icpe-trx）25–44 MiB／job，Mooncake 幾乎 0。
- 修好、G1 過了，我 merge → G0 → 重開 `tl_main`（c1 從頭跑）。

**時程影響**〔推算〕：TraceLab 從修正 merge 後才重新開始，約 20–28 h。在那之前，40 worker 先做其他非 TL 的工作，總工作量不變。

## 2026-11-05T09:10+08:00 — npz 指紋改成不含 `meta_json`（第一版作廢）；SemiAnalysis 13／13 和 TRX40 相同

- **錯在我**：`digest` 第一版把 npz 裡的 `meta_json` 陣列也算進去。它存 run_id、ts、host 和絕對路徑，換一台機器、甚至同一台重轉一次都會不同（icpe-trx 在 STATUS_TRX 09:07 指出）。
- 修正：`cmd_digest` 跳過 `meta_json`，只算 8 個資料陣列（`full_key`、`full_off`、`kind`、`nall`、`parent`、`sess`、`t`、`user_trigger`）。
- 重算：run `20261105-090851-icpe-data-npz-digest-nometa`，**取代** `results/icpe_next/b/traces/npz_digest.csv`。欄名改成 `content_sha256_no_meta`；第一版 run `20261105-084835-icpe-data-npz-digest` 作廢。
- **和 TRX40 比**（`origin/icpe-next-trx` 652f943 的 STATUS_TRX 表，run `20261105-090355-icpe-trx-npz-digest`）：**SemiAnalysis 13／13 全部相同**（含 N16 s1 t128k）。
  - 資料陣列逐位元相同。`meta_json` 的彙總欄位 icpe-trx 也比過，13／13 相同。
  - → icpe-trx 可以照它的使用者決定往下做：xc_sa，然後 sa_main 用 40 worker。

## 2026-11-05T09:05+08:00 — H200 不能接 → 舊紀錄＋Bailian 回 acvlab（第二條佇列 4 worker）；PREREG 偏離 5；CacheWise 最終上限表

**分工更新**：h200 回報 H200 不能送純 CPU 的 Slurm job（STATUS_A 728cf82：h200q 會把每個 job 改成 1 GPU＋16 核＋170 GB，帳號的 4 張 GPU 已滿）。所以上一節表裡 H200 那一列**全部回到 acvlab**：`old_main`、`bl_main`、兩者的 `_ttl7`、`sens_rep` 的 `mc_*／fi_*／bl_*`。TRX40（icpe-trx）不變。

**做完什麼**
- **acvlab 第二條佇列**（使用者 11/5：先同意多開 8 個 worker，後改成「太多 改 4」）：`run --plans old_main bl_main --workers 4`，log `/ssd6/hungwei/icpe-next/runs/queue-old_bl-w4-20261105-085744.log`。
  - 8 worker 的版本只跑了約 1 分鐘就停掉，run `20261105-085649-icpe-data-old_main-mooncake` 留了 `ABORTED` 檔。
  - TraceLab 那條（40 worker）沒有動。現在我們共 44 個 worker。
  - load 監看：連續 10 分鐘 >100 就通知我（`load-watch-20261105.log`）。
- **merge**：
  - `origin/icpe-next-sim` @ 9acfec1 → 94b5ec8。只改 `next_sim_analyze.py`，加了 `--rival-only-gpu`。
  - `origin/icpe-next-a` @ 20fe61a → 9af6d04。PREREG 和 h200 自己的檔，和本分支的檔沒有重疊。
  - 模擬程式（`code/p3x`、`p3_fvar.py`、`p3_platform.py`）和 7539f01 **逐位元相同**（`git diff` 是空的），所以沒有重做 G0。
- **PREREG 偏離 5**（commit **8f362d8**，`date -u` 2026-11-05T00:59:15Z）：偏離 4 第 2 項的釐清——補跑只擴充 ② 的 GPU 規則，④ 維持 3 個 TTL。使用者在 h200 選「交給 acvlab 寫」（STATUS_A 20fe61a）。節內照實列出寫的當下我已經看過的正式結果。
- **CacheWise 最終上限表**：run `20261105-085942-icpe-data-analyze-cw_main-final`（cw_main＋cw_main_ttl7，`--rival-only-gpu TTLA/TTLB:1/3/30/100`）→ `results/icpe_next/b/analysis/cw_main_final/`。
  - 3,888 列，格子集合和補跑前相同。
  - **0 列上限變大**、213 列變小。
  - violations 369 列：② 多了規則，「④ 比某個 ② 組合慢」的列跟著變多。

**需要 h200 做什麼**：PREREG 偏離 5 在 `icpe-next-data` 8f362d8。要不要 merge 進 `icpe-next-a` 由你決定；本分支已經 merge 過 20fe61a，所以只差這一節。

## 2026-11-05T09:00+08:00 — CacheWise 補跑完成；TraceLab 改 40 worker；**三台分工（使用者 11/5 決定）**；分析程式要補一個選項

**做完什麼**
- **`cw_main_ttl7` 17 個 block 全部 exit 0**（07:24 → 08:40，32 worker），19,840 列，已收進 `results/icpe_next/b/cw_main_ttl7/`。
- **TraceLab 改 40 worker**（使用者 11/5：「改大一點 到 40 加速？」）。
  - 量測：7539f01 的 TraceLab c1，32 worker 開跑後每個 worker 的 PSS 0.55–0.97 GiB，合計 20.4 GiB（parent 310 MiB）。load 約 67／96。
  - 處理：c1 才跑 5 分鐘，就把整個佇列停掉（kill 行程群組）。run `20261105-084040-icpe-data-tl_main-tl_c1` 留了 `ABORTED` 檔說明原因，沒有收任何列。
  - 重開：`run --plans tl_main --workers 40`，log `/ssd6/hungwei/icpe-next/runs/queue-tl_main-w40-20261105-084551.log`。記憶體監看在 `mem-watch-queue-tl_main-w40.log`。
- **CacheWise 補跑後的分析**（run `20261105-084414-icpe-data-analyze-cw_main-ttl7`）發現一個問題，見下面「需要 icpe-sim 做什麼」。在它修好前，這張表不進 results/。
- 佇列程式（本節 commit）：
  - 路徑可用環境變數設定：`NEXT_B_ROOT`、`NEXT_B_PY`、`NEXT_B_NB_DIR`、`NEXT_B_MOONCAKE`、`NEXT_B_FI_DIR`、`NEXT_B_TAG`。
  - `run --only '<glob>'`：只跑名稱符合的 block。
  - plan `xcheck`：跨機器對拍，新功能開啟、小格子。`xc_sa`＝SemiAnalysis N16 s1、`xc_bl`＝Bailian coder；CPU 64、LRU／GDSF／O-BEL／O-PF × GPU LRU／BEL／TTLA:10／TTLB:10、ssd、cake。
  - `digest`：紀錄 npz 的內容指紋。`savez_compressed` 的 zip 有時間戳，檔案的 sha256 每次轉換都不同，所以比內容。
  - `next_b_g0.py`：也去掉 `host` 欄；有新功能的欄時也加進鍵；可讀 .xz。舊 G0 的比法不變：對 `20261105-070841` 的輸出重比，仍是 128,064 格 0 個不同。
- **紀錄內容指紋**：run `20261105-084835-icpe-data-npz-digest` → `results/icpe_next/b/traces/npz_digest.csv`。涵蓋 SemiAnalysis 12＋N16 s1 t128k、Bailian 3、FI 第 1 週 t128k。

**三台分工**（使用者 11/5 在 icpe-data 決定；NEXT_PLAN §3b「正式模擬都由 icpe-data 跑」因此改成下表）

| 機器 | session | 跑什麼 |
|:--|:--|:--|
| acvlab | icpe-data | TraceLab：`tl_main` → `tl_main_ttl7`<br>CacheWise／TraceLab 的敏感度：`sens_rep --only 'cw_*' 'tl_*'`<br>CacheWise 沒有授權，**只能在 acvlab** |
| TRX40（Threadripper 3960X 48 執行緒、62 GiB；不碰 GPU） | **icpe-trx**（新開） | SemiAnalysis：`sa_main` → `sa_main_ttl7` → `sens_rep --only 'sa_*'` |
| H200 | h200 | 若 Slurm 能送純 CPU 的 job：`old_main` → `bl_main` → 兩者的 `_ttl7` → `sens_rep --only 'mc_*' 'fi_*' 'bl_*'`<br>不能的話留給 acvlab |

授權：SemiAnalysis（Apache-2.0）、Bailian（Apache-2.0）、TraceLab（資料 CC BY 4.0）可以放別台；**CacheWise 不行**。

### 給 `icpe-trx`（TRX40）的步驟

任何一步失敗：記完整錯誤，寫進 STATUS_TRX，停下來。

0. **Repo**：clone `https://github.com/s990093/paper-hierarchical-kv-state`，從 `origin/icpe-next-data` 的最新 commit 開分支 **`icpe-next-trx`**，push 那個分支。
   - 只寫 `docs/icpe_brief_20261030/next/STATUS_TRX.md`（新檔，格式同本檔）和 `results/icpe_next/b/sa_*`、`results/icpe_next/b/analysis/sa_*`、`results/icpe_next/b/sens_rep/sa_*`。
   - 不改任何程式。要改 `next_b_*` 就寫在 STATUS_TRX，由 icpe-data 改。
   - 根目錄 `$ROOT` 放在一顆大磁碟上（例如 `/ssd4/hungwei/icpe-next`，先看 `df -h`；/ssd2 已 91%）。底下分 `venv/`、`uv-python/`、`datasets/`（下載）、`runs/`、`hf-home/`；程式在 repo 裡，和資料分開。
1. **venv**：Python **3.12.15**、numpy **2.3.5**、pandas **3.0.5**、pyarrow **25.0.1**（相依 python-dateutil 4.0.0、tzdata 2026.6），和 acvlab 完全相同。
   - 本機的 uv 裝不到 3.12.15 時：`uvx --from uv==0.13.7 uv python install 3.12.15 --install-dir $ROOT/uv-python --no-bin`。
   - 把版本寫進 STATUS_TRX。
2. **下載**（不受信任的資料：只讀 JSON／parquet、Python 一律 `-I`、資料和程式不同目錄）：
   - `code/setup_3090_data.sh semianalysis`：HF revision 23f152f6，`hf_verify files=3 bad=0`。acvlab 的做法見 run `20261104-233022-icpe-data-dl-tracelab-semianalysis` 的環境變數（`TIARA_HOME`、`TIARA_DATASETS`、`HF_HOME`、`TIARA_HF_CLI`）。
   - `code/setup_3090_data.sh mooncake`：commit d750cdda，conversation sha256 前 8 碼 `b8cbb061`、toolagent `48a2db1a`。
3. **轉換**：
   - p3x 用 `git archive 6b17191 code/p3x`，和 acvlab 轉換時同一版。
   - 指令：`TIARA_DATASETS=$ROOT/datasets python -I code/icpe/next_b_convert.py semianalysis --N 16 32 64 72 --seeds 1 2 3 --out-dir $ROOT/datasets/x1_nb --p3x <archive>/code/p3x`
   - 截斷版：同一支，`semianalysis --N 16 --seeds 1 --trunc-128k ...`。
   - 驗證：`next_b_queue.py digest <13 個名字>` 要和 `results/icpe_next/b/traces/npz_digest.csv` **完全相同**。不同就停。
4. **G0**：照 acvlab run `20261105-070841-icpe-data-g0-merged-7539f01` 的 `cmd.sh`（換成本機路徑、16 worker），對 `results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz` 用 `next_b_g0.py` 比，要 `G0_RESULT IDENTICAL`。
5. **跨機器對拍（新功能）**：
   - 環境：`NEXT_B_ROOT=$ROOT NEXT_B_PY=$ROOT/venv/bin/python NEXT_B_MOONCAKE=<mooncake traces> NEXT_B_TAG=icpe-trx`。
   - 跑：`python -I code/icpe/next_b_queue.py run --plans xcheck --only xc_sa --workers 16`。
   - 比：`next_b_g0.py results/icpe_next/b/xcheck/xc_sa/x1_sweep.csv.xz $ROOT/runs/<那個 run>/out/x1/x1_sweep.csv`，要 IDENTICAL。
   - **不要** collect xcheck。
6. **正式**（同一組環境變數）：
   - `run --plans sa_main --workers 16`。
   - 第一個 block 開跑 5 分鐘後，量每個 worker 的 PSS（`/proc/<pid>/smaps_rollup`）和 `free -g`。若 MemAvailable 保持 >12 GiB，就在 block 之間加 worker，最多 40。不能用到 swap。nice -n 10 已內建。
   - 每個 block 跑完：`collect --plans sa_main` → commit `results(icpe-next-b): ...` → push `icpe-next-trx`。
   - `sa_main` 12 個都完成後：
     1. `RUN_ID=<run_id> python -I code/icpe/next_sim_analyze.py --sweep results/icpe_next/b/sa_main/*/x1_sweep.csv.xz --out-dir <run>/out --out-prefix sa_main_ttl3 --pf-target cpu`（§4.1 的殼，一定要 export RUN_ID）。
     2. `ttl7cells --plan sa_main --headroom <run>/out/sa_main_ttl3_headroom_components.csv`。
     3. 表（xz）放 `results/icpe_next/b/analysis/sa_main_ttl3/`，`cells.csv` 一起 commit。
     4. `run --plans sa_main_ttl7` → collect。
     5. `run --plans sens_rep --only 'sa_*'` → collect。
   - **不解讀上限、不下判定**。最後的分析由 icpe-data 做。
7. **規則**：
   - CLAUDE.md 的不可協商規則：不編數字（沒跑寫 NOT_MEASURED）、失敗就記完整錯誤並停、每個數字追溯到指令與輸出檔、不改 main.tex。
   - 不改 `code/p3x/*`、NEXT_PLAN、其他 STATUS 檔。
   - 不用 GPU。CacheWise 的任何東西都不放 TRX40。
   - git：不 stash、不 rebase、不合回 main；commit 結尾加 attribution。

### 給 `h200` 的請求

1. **先確認**：Slurm 能不能送純 CPU 的 job？每個節點幾核、多少記憶體、時間上限多少？可以的話，照上表接 `old_main`、`bl_main`。
   - 步驟同 icpe-trx 的 0–7。差別一：分支、STATUS 檔用你自己的。差別二：資料。
   - 資料：Mooncake、FI 你那邊已經有。Bailian 要下載（`setup_3090_data.sh bailian`，commit 5f7439c5）並轉換：`bailian --files traceA thinking coder`。FI 第 1 週要轉截斷版：`fi --weeks 2026-05-17_2026-05-23 --trunc-128k`。
   - 指紋要和 `npz_digest.csv` 相同。xcheck 用 `xc_bl`。
   - 不能送就回一句，我留在 acvlab 跑。
2. **PREREG 釐清**（偏離 4 第 2 項的補跑規則）：見下一項。請寫一句釐清：「補跑 ② 時，④ 的 GPU 規則集合維持原本 3 個 TTL」。這正是 h200 註解「② 只會隨 TTL 值變多而變快」的前提；如果 ④ 也加 TTL，上限可能變大，這個論證就不成立。

**需要 icpe-sim 做什麼**（不擋模擬，只擋 CacheWise 的最終表）
- 問題：`next_sim_analyze.py` 對 4 個 CPU 元件（O-ORD／O-DROP／O-NOW／O-PF）的定義是 ④＝COMPONENT_POLICIES × {GPU 不預測規則}、②＝RIVALS × {GPU 不預測規則}，兩邊用**同一組** GPU 規則。
- 偏離 4 的補跑只加在 ②。所以補跑過的 248 格裡，④ 缺新 TTL 的列，程式把這 4 個元件**整列不算**：2,896 列，補跑前是 3,888 列。O-GPU、O-ALL(report) 不受影響。
- 請加一個選項，例如 `--rival-only-gpu TTLA:1 TTLA:3 TTLA:30 TTLA:100 TTLB:1 TTLB:3 TTLB:30 TTLB:100`：這些 GPU 規則只放進 ②（rset），不放進 ④（oset）。沒補跑的格子沒有這些列，結果和現在一樣。
- 檢查：開這個選項時，任何格子的上限都不能比 3 個 TTL 時大。我在 CacheWise 上會驗。
- 程式是你的，我不改。

**需要使用者決定什麼**：無（分工已決定）。

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
