# STATUS_RUX（icpe-rux，RUX3DPRO）

本檔由 icpe-rux 寫，最新的一節在最上面。格式照 `STATUS_TRX.md`。
**到 11/8 01:00：正式模擬只有使用者同意、登記過的（PREREG 偏離 10、11、19）；沒有動佇列；別人的檔只有 PREREG 檔尾 append（使用者授權）。偏離 14 已移植成 v8_dev14（36780534）。偏離 19 完成並 push（7cb8ebdd）；機器閒置，W7 照指示停著。**

---

## 2026-11-08T01:00+08:00 — 偏離 19（二階排隊）12 個變體完成：結果與加總表 push（7cb8ebdd）

**問題／要證明什麼**：排隊的回饋算進去之後（快取照服務順序演進），版本 C 的上限在輕、中負載會不會變？和一階的差別是到達規則還是回饋？屬於敏感度，主結果（版本 C、不含排隊）不變。

**結果**（`results/icpe_next/queue2/aggregate_SA12/agg_summary.md`；12 個變體 × 54 格＝648 格，cake；負載各自用自己那一格 (LRU, LRU) 的平均等待分級，重負載只報告）
- 情境：
  - 11B＝偏離 11 B 一階（沒有 pfx，基準＝舊版本 C）；
  - 1m＝一階、到達不早於上一個完成（pfx=1）；
  - 2nd＝二階（pfx=1）。
  - 1m、2nd 的基準都是修正版版本 C（icpe-next-trx e47af649）。
- **負載分布**：輕負載只有 N16；中負載幾乎都在 N16、N32（二階 136 格：N16 70、N32 65、N64 1）；N64、N72 幾乎全是重負載（只報告）。
- **比較 1：二階 vs 修正版版本 C**
  - 輕（56 格）：沒有任何一格 ≥ 15%。O-GPU 差的中位數 −3.31 個百分點。
  - 中（136 格）：
    - O-GPU ≥ 15% 的格數 17 → 2（16 格降、1 格升），中位數 −4.25 個百分點；
    - O-ALL(report) 21 → 9；O-ORD 0 → 1；O-PF 2 → 6（5 升、1 降）。
  - 重（456 格，只報告）：O-GPU 304 → 25；O-PF 17 → 134；O-ORD 161 → 73。
- **一階本身 vs 自己的基準（中負載）**：
  - 1m（153 格）：O-GPU 10 → 39、O-ORD 1 → 18、O-PF 0 → 12；
  - 11B（154 格）：O-GPU 9 → 34、O-ORD 1 → 14、O-PF 0 → 10；
  - 輕負載兩者都是 0 格。
- **比較 2：回饋（2nd vs 1m，依二階的負載）**
  - 輕：O-GPU 2 → 0，中位數 −2.79 個百分點。
  - 中：
    - O-GPU 38 → 2（37 格降、1 格升），中位數 −6.76 個百分點（p10 −39.5、p90 −0.9）；
    - O-ORD 11 → 1；O-PF 17 → 6（13 降、2 升）；O-ALL(report) 41 → 9。
- **到達規則（＋pfx）：1m vs 11B（依 1m 的負載）**
  - 非預取元件（只差到達規則）：每個負載等級的中位數都在 −0.18～+0.54 個百分點；≥ 15% 的格數幾乎不變（中負載 O-GPU 37 → 39、O-ORD 18 → 18）。
  - 預取元件（另外有 pfx）：中負載 O-ALL(report) 中位數 +1.36、O-PF 11 → 12；重負載 O-PF +4.07。
- **負載等級交叉（2nd × 1m）**：
  - 輕 52、中 127、重 434 格一致；
  - 4 格二階輕、一階中；22 格二階重、一階中；8 格二階中、一階重。
- **重負載的平均等待最大值**：11B 52,033 s、1m 51,941 s、2nd 3,210 s。一階的排隊事後套上，等待會一直累積。
- 〔推論〕
  - 一階在中負載看到的上限放大（O-GPU 10 → 39），大部分是「快取不受排隊影響」造成的；回饋算進去之後幾乎消失。
  - 到達規則本身影響很小。
  - 二階的下降有一部分可能來自 ④ 只知道紀錄順序（登記寫明的偏低方向）。
  - 這些結論只建立在 N16、N32 上。

**檢查**
- 12 個變體都是 `QUEUE_RESULT E2E_OK rows=324`、`QUEUE2_RESULT E2E_OK rows=324`。
- 加總表 `AGG_RESULT OK variants=12 rows=3888`（run `20261108-004610-icpe-rux-q2-agg-SA12`）：
  - 12 個變體的非預取元件，舊、修正版版本 C 逐值相同；
  - 預取元件有 84–108／108 格不同（偏離 14 只改這兩列）。
- 基準 e47af649：12 個變體的 `verC_headroom_components.csv`、`x1_sweep.csv` 照 MANIFEST 驗 sha256、bytes、列數，全部相符，code_sha `3280b1c8799b+rs4eca5bda`。N16_s1 是用 33a20f38 分析的，兩個 commit 的檔案完全相同。
- 所有 sweep 的 code_sha 都是 `3280b1c8799b+rs4d1cc219`（v9 9496c4f6），`py_fallback_jobs=0`。

**經過**
- 11/7 19:57 WSL 整個重啟，chain 被殺，scratchpad 被清空。
  - 已完成的 6 個變體（N16、N32）逐一確認後保留。
  - N64_s1（run `193501`）在 B 段中斷，標 `INTERRUPTED.txt` 作廢，用新 run `200316` 整個重跑（偏離 19 §5）。
- chain 腳本移進 repo（`v9_qev/run_queue2_chain.sh`、`run_queue2_chain_analyze.sh`，f7c83f77、d08c6147）：已完成的跳過，每個變體檢查 code_sha。
- 加總表 `agg_queue2.py` 和收尾 `run_queue2_finalize.sh`（61b97c97）。
- 使用者要求降負載：21:18 起 chain 限制在 CPU 0–13，21:31 起 0–9（taskset；worker 數照登記仍是 16，只影響速度）。紀錄在 N64_s2 run 的 `AFFINITY.txt`。
- sweep 13 個 run，最後一個 11/8 00:45 完成；分析和加總表 00:46 完成。

**接下來**
- 偏離 19 做完。論文照使用者 11/7 約 17:1x 的寫法三者並列：
  - 11B → 1m 標「到達規則＋pfx」，預取元件分開；
  - 1m → 2nd 標「回饋」；
  - 註明 11B 沒有偏離 14 的修正。
- W7 照使用者的指示停著。沒有其他排定的工作。

---

## 2026-11-07T17:20+08:00 — 偏離 19（二階排隊）登記並 push（6da749c4），12 個變體的模擬開始

**要證明什麼**：排隊的回饋算進去之後（GPU 照服務順序處理、快取照服務順序演進），版本 C 的上限在輕、中負載的格會不會變。屬於敏感度，主結果不變。

**使用者的決定**：看過登記稿之後（AskUserQuestion，11/7 約 17:15）選「push 並開跑」。之前的決定見上一節。

**做了什麼**
- **引擎與驗證**：
  - v9 改成從 v8_dev14 產生（commit 9496c4f6，main.rs sha256 `8fcdab56…`）。
  - 二階下偏離 14 (i) 的「已經過去」改成「已經服務過」。
  - 重驗（run `20261107-163027-icpe-rux-v9c-smoke`）：不開旗標、每個 job 開 `pfx=1`，兩輪各 2,856／2,856。
- **v8_dev14 補驗**：D3（整個 session 的 APC＝LRU、BELP＝BEL，加雜訊與三個旗標）在 Bailian coder、Bailian thinking 各 408／408（run `20261107-165928-icpe-rux-d14-d3-bailian`）。
- **流程**：
  - 拆成 `run_queue2_sweep.sh`（A：一階 `qpost=2`；B：二階 `qev`；都開 `pfx=1`、`pgr=1`）和 `run_queue2_analyze.sh`（和修正版版本 C 比、收集）。
  - 使用者 11/7 約 16:3x 決定第二段版本 C 全部用 v8_dev14 重跑，所以比較的基準等 TRX40 的修正版 SemiAnalysis 版本 C。
  - 中間寫過一版「只補 O-PF／O-ALL」的 C 段，沒跑就拿掉了。
- **端到端試跑**：N16 s1、111.0625 GiB（run `20261107-170242`／`170338`／`170553`）。
  - 基準用 v8_dev14 自己全部重跑。
  - 兩段 E2E_OK，MANIFEST 對到 v9（9496c4f6）。
  - 只看標記行。
  - 時間：每個容量 A 段 51 s、B 段 82 s。
- **PREREG**：偏離 19 append 93 行、0 行刪除（6da749c4）。commit 已給 h200 同步，也通知了 icpe-main。

**接下來**
- 模擬鏈：12 個變體依序跑 `run_queue2_sweep.sh`，約 5–6 小時〔推算〕。
- TRX40 的修正版 SemiAnalysis 版本 C 出來之後跑 `run_queue2_analyze.sh`，收集、push、回報。
- W7 照使用者的指示停著。

---

## 2026-11-07T16:30+08:00 — 偏離 14 移植成 v8_dev14（36780534），驗證通過；二階排隊（偏離 19）的引擎和驗證完成，正式 run 排在移植之後

**使用者的決定**（照時間）
- 約 15:0x，在 icpe-main：「让那个机器跑二阶事件」（icpe-main 轉達）。
- 約 15:12，在 icpe-rux 用 AskUserQuestion：④ 的順序選「紀錄順序」、範圍選「SemiAnalysis 12 個全部」、重負載選「照偏離 11：只報告」。
- 約 15:40，在 icpe-rux：到達規則選「二階改，一階也重跑」。
- 約 15:4x，在 icpe-main：「先移植 v8，再做二階排隊（建議）」（icpe-main 轉達）。
- h200 給偏離 19。登記稿寫好了，還沒 push，要先給使用者看。

**A. 偏離 14 → v8_dev14**（`code/icpe/rux_x1fast/v8_dev14/`，`rux/V8_DEV14.md`）
- 要證明什麼：一個引擎同時跑雜訊、偏離 14 的規則、APC／BELP。修正版版本 C 的 O-PF／O-ALL 列和 P4 加跑要用它。
- `make_v8_dev14.py` 從 v8（48e4efce）產生，v8 不動。來源是 icpe-sim 的 x1fast 63b8e328（40a4c87f 驗證過），對 4a4baa48 的差異。
- **給第二段的資訊**（icpe-data 要的三樣）：
  - commit：`36780534`（origin/icpe-next-rux）。
  - `v8_dev14/src/main.rs` sha256 `d2048fa38a42c78b316a97734101cb094854cc584be92ef5ec9c8717c6ad5e88`。
  - sweep 的 src sha（Cargo.toml＋main.rs）`4eca5bda266e007b68a0fbf60cdb5fbb4d0d0a693bfcca5b4566e3fb87eb6db9`，CSV 的 code_sha `3280b1c8799b+rs4eca5bda`。
  - 旗標名稱和 x1fast 相同：jobs.txt `pfx=1`、`nzo=resample`、`pgr=1`；CSV `pf_expire`、`noise_overdue`、`pf_gpu_recheck`。
  - 本分支的 x1_sim 沒有命令列參數，`sweep_v8_dev14.py` 用環境變數 `RUX_PFX=1`、`RUX_NZO=resample`、`RUX_PGR=1` 開。
- **APC／BELP 和規則的交互作用**：
  - v8 不准 APC／BELP 預取進 GPU，所以 `pf_expire` 的 GPU 路徑和 `pgr` 碰不到 APC 的部分副本，pfx 只走 CPU 路徑（和 x1fast 同一段程式）。
  - BELP 經 `nz()` 讀雜訊，所以 `nzo=resample` 也作用在 BELP。
- **驗證**（SemiAnalysis N16 s1；CPU 16、111.0625 GiB × 17 個 CPU 規則）：
  - 旗標全關：和 v8 48e4efce 逐位元相同，564／564。
  - 不開 APC／BELP、開旗標：和 x1fast 63b8e328 逐位元相同 1,908 組。另 2 組兩邊都斷言，panic 訊息相同（`gspf 363 56`、`gspf 911 4`）。條件是只開 pfx、沒開 pgr、O-ALL、預取到 GPU、雜訊 P B2，這是參考版本身的行為；三個旗標全開時同一格跑得完。
  - 整個 session 的 APC＝LRU、BELP＝BEL（含雜訊與三個旗標）：204／204；逐塊逐出加稽核 204／204。
  - wt（只預取到 CPU）：2,414／2,414。
  - sweep 端到端：開旗標只有 O-PF、O-ALL 的列改變。
  - run：`20261107-155514-icpe-rux-d14-smoke-full`、`161328-…-d14-smoke-oall`、`161521-…-d14-smoke-wt`、`162547-…-d14-sweep-e2e`。
- icpe-sim 會用它的 t1n／t1q／T2／xref_dev14 再交叉檢查一次。

**B. 二階排隊（偏離 19）**（`code/icpe/rux_x1fast/v9_qev/`，`rux/QUEUE2_DESIGN.md`；還沒 commit）
- 要證明什麼：排隊的回饋算進去之後（GPU 照服務順序處理、快取照服務順序演進），版本 C 的上限在輕、中負載的格會不會變。屬於敏感度，主結果不變。
- v9＝`make_v9.py` 從 v8_dev14 產生，加 `qev=1`（二階）、`qpost=2`（一階、到達不早於上一個完成）。
- 驗證（在 v8 基底上）：v9 不開 qev＝基底、到達拉開＝沒排隊、Python 精確重算排程、稽核、決定性、`qpost=2` 精確重算，2,856／2,856 通過。另有獨立唯讀審查，沒有確定的錯誤。
  - 換到 v8_dev14 基底之後要重跑。
- **審查找到的到達問題**：原規則允許請求早於同 session 上一個請求完成就到達。量起來輕負載約 2% 的請求，卻占總等待 16–21%。使用者選「二階改，一階也重跑」。偏離 11 B 的一階結果有同樣的高估，結果不改。
- **接下來**：
  1. v9 在 v8_dev14 基底上重驗。
  2. 正式流程加修正版版本 C（pfx）的基準。
  3. 登記稿給使用者看，push 後把 commit 給 h200。
  4. 正式 run（約 12–16 小時〔推算〕）。

---

## 2026-11-07T14:30+08:00 — 偏離 11 的正式 W5（另外 3 個變體）、排隊（SemiAnalysis 12 個）跑完；兩者都只報告，主結果不變

**要證明什麼**
- W5：換成 W5 格式（allowed＝每組事後挑較快的格式）之後，版本 C 的上限會不會變；升高超過 5 個百分點的格要寫機制（NP:53）。
- 排隊：把排隊等待算進 TTFT（qpost）之後，同一格的上限會不會變。依 (LRU, LRU) 的平均等待分成輕 <2 s、中 2–20 s、重 ≥20 s，重的只報告。

**做了什麼**
- `chain_w5q.sh` 在 `CHAIN_APC_OK` 之後自動開始，12:08–14:18，印 `CHAIN_W5Q_OK`。
- 每個 run 都是 `exit=0`、`py_fallback_jobs=0`，引擎 v8 48e4efce，sweep code_sha `3280b1c8799b+rse7f6ee8f`（MANIFEST 的 engine_used 只有這一個）。
- **W5**（run `20261107-120835`／`121216`／`121326`）：Bailian coder、Mooncake conversation R1、toolagent R1。
  - 每個變體 4,728 個 job，BF16 和 W5 各 5,508 列。
  - W5 較快的組合：Bailian coder 5,506、conversation 5,480、toolagent 5,503（各 5,508 組）。
  - **核對**：每個變體的 `base` 上限表和版本 C 同 3 個容量的 108 個 (格, 元件)，7 欄都是 0 個不同（SemiAnalysis N16 s1 也重新核過）。
- **排隊**（run `20261107-121712` 到 `140650`）：SemiAnalysis 12 個 × 9 個容量。
  - 每個都印 `QUEUE_RESULT E2E_OK rows=324`，也就是只有 TTFT 推出來的欄位不同。

**W5 結果**（cake，每個變體 18 格；≥15% 的格數，版本 C → allowed；差＝allowed − 版本 C 的中位數，單位百分點）

| 元件 | SemiAnalysis N16 s1 | Bailian coder | Mooncake conv R1 | Mooncake tool R1 |
|:--|:--|:--|:--|:--|
| O-GPU | 2 → 0（−2.36） | 8 → 3（−3.86） | 18 → 13（−4.97） | 18 → 8（−8.43） |
| O-ORD | 2 → 0（−2.19） | 6 → 6（−1.09） | 18 → 18（+8.10） | 17 → 14（+2.24） |
| O-PF | 0 → 0（−1.95） | 8 → 6（−5.16） | 18 → 18（−0.95） | 18 → 14（−5.06） |
| O-DROP | 0 → 0（−0.20） | 6 → 6（−0.37） | 18 → 18（+5.88） | 18 → 16（+0.13） |
| O-NOW | 0 → 0（+0.13） | 0 → 0（+0.01） | 0 → 0（+0.13） | 0 → 0（0.00） |

- **沒有任何格因為 W5 從 <15% 變成 ≥15%**。反方向的有：
  - Bailian coder：O-GPU 5 格、O-PF 2 格、O-ALL(report) 4 格。
  - conversation：O-GPU 5 格。
  - toolagent：O-GPU 10 格、O-PF 4 格、O-ORD 3 格、O-DROP 2 格。
- **升高 >5 個百分點的 (格, 元件)**（`w5_mechanism_rows.csv`）：
  - SemiAnalysis N16 s1：3 個，全部在 nfs（上一節）。
  - Bailian coder：1 個，O-DROP nfs，+5.02，上限 0.197 → 0.247。
  - Mooncake conversation：34 個，O-ORD 12、O-DROP 10、O-PF 6、O-NOW 6。
  - Mooncake toolagent：20 個，O-NOW 6、O-ORD 6、O-DROP 4、O-PF 4。
- **機制**（兩邊都用 W5 格式）：
  - O-ORD、O-DROP、O-PF 這 3 個元件：oracle 的慢層載入少得比對手多。
    - 各元件 oracle 減少量的中位數：Mooncake −68%～−99%，Bailian −62%；對手 −14%～−29%。
    - 重算也是 oracle 少得多。
    - 所以 W5 多出來的容量，oracle 用得比對手好。
    - 這些格本來就 ≥15%，W5 之後更高：O-ORD 0.166–0.494 → 0.240–0.696，O-PF 0.283–0.414 → 0.498–0.579。
  - O-NOW 的 12 格是負值變得沒那麼負：−0.352～−0.124 → −0.180～+0.040，都還 <15%。兩邊慢層載入少得差不多（−23%～−32%）。
- 結果在 `results/icpe_next/w5/<變體>/`（含 SemiAnalysis N16 s1 共 4 個）。TRX40 的 TraceLab c1、FreeInference w1 在 `origin/icpe-next-trx`。

**排隊結果**（cake，12 個變體 × 54 格 × 6 元件＝3,888 列）
- 各負載的格數：輕 282 列（只有 N16）、中 924 列（N16、N32、N64 s2）、重 2,682 列（N64、N72 幾乎全部）。

| 負載 | 元件 | ≥15%（不含 → 含排隊） | 跨過 15%（升／降） | 差的中位數（百分點） |
|:--|:--|:--|:--|--:|
| 輕（47 格） | O-GPU／O-ORD／O-PF／O-DROP／O-NOW | 全部 0 → 0 | 0／0 | −0.73～+0.07 |
| 中（154 格） | O-GPU | 9 → 34 | 25／0 | +1.32 |
| 中 | O-ORD | 1 → 14 | 13／0 | +0.86 |
| 中 | O-PF | 0 → 10 | 10／0 | −0.36 |
| 中 | O-DROP、O-NOW | 0 → 0 | 0／0 | +0.12、−0.33 |
| 重（447 格，只報告） | O-GPU | 312 → 422 | 117／7 | +20.24 |
| 重 | O-ORD | 160 → 326 | 166／0 | +17.19 |
| 重 | O-PF | 10 → 54 | 44／0 | −0.37 |
| 重 | O-DROP | 0 → 2 | 2／0 | +0.41 |
| 重 | O-NOW | 0 → 0 | 0／0 | −0.48 |

- 中負載跨上 15% 的：
  - O-GPU 25 格（N16 12、N32 11、N64 2）。
  - O-ORD 13 格（N16 6、N32 7）。
  - O-PF 10 格（全部 N32）。
  - 這些格 (LRU, LRU) 的平均等待中位數 9–13 s。
- 讀法〔推論〕：服務時間短一點，排隊等待就短更多，所以含排隊時相對上限變大。
  - 輕負載（<2 s）：差的絕對值 >5 個百分點的只有 O-PF 1 格。
  - 中負載：有影響。
  - 重負載：等待是幾百到幾千秒，而 qpost 是一階的事後 FCFS、不是事件驅動的閉迴路，所以不可靠，只報告。
- 依登記：主結果維持版本 C、標「不含排隊」；排隊是敏感度。
- 結果在 `results/icpe_next/queue/<變體>/`（`queue_rows.csv`、`queue_headroom_components.csv`、`x1_sweep.csv.xz`、MANIFEST）。

**接下來**
- RUX3DPRO 這邊偏離 10、11 都跑完了。機器現在空著。
- W7 照使用者指示停著。
- 可以接的：UltraChat（如果 icpe-data 要備援）、BELP 加速、二階排隊（事件驅動）。

---

## 2026-11-07T12:10+08:00 — 偏離 10 的正式 APC 重算：RUX3DPRO 這邊 17 個變體全部跑完，版本 B 全部逐值重現（dcc8e00c）；W5／排隊鏈已自動開始

**要證明什麼**：加 APC（② ④）和 BELP（只進 O-GPU ④）之後的版本 C 上限表。前提是只用版本 B 的 16 個 GPU 規則時，能逐值重現版本 B。

**做了什麼**
- `run_apc_formal.sh` 依序跑完 SemiAnalysis 12 個、Mooncake conversation／toolagent、Bailian coder／thinking／traceA（thinking、traceA 先轉成 nb，run `20261107-115310-icpe-rux-convert-bailian-thinking-traceA`）。鏈結尾印 `CHAIN_APC_OK`。
- 每個 run 都是 `exit=0`、`py_fallback_jobs=0`，引擎 v8 48e4efce，sweep code_sha `3280b1c8799b+rse7f6ee8f`（MANIFEST 的 engine_used 只有這一個）。
- **重現檢查**：17 個都是 `EXPLORE_RESULT REPRODUCED`，7 欄全部 0 個不同。SemiAnalysis、Bailian 每個 324 個 (格, 元件)；Mooncake 每個 918 個。
- 結果在 `results/icpe_next/apc/<變體>/`：f9717083、d35aed35、f054a1aa、dcc8e00c。

**版本 B → 版本 C**（cake；≥15% 的格數，各家族加總；括號內是單格上限變化的範圍）

| 元件 | SemiAnalysis（12 × 54 格） | Mooncake（2 × 162 格） | Bailian（3 × 54 格） |
|:--|:--|:--|:--|
| O-GPU | 333 → 321（−0.214～+0.122） | 285 → 287（−0.0001～+0.010） | 67 → 67（−0.0003～+0.009） |
| O-ORD | 183 → 161（−0.054～+0.010） | 212 → 212 | 26 → 26 |
| O-PF | 19 → 10（−0.091～+0.910＊） | 272 → 272 | 42 → 42 |
| O-DROP | 0 → 0 | 228 → 228 | 40 → 40 |
| O-NOW | 0 → 0 | 0 → 0 | 0 → 0 |
| O-ALL(report) | 314 → 268 | 216 → 216（2 × 108 格） | 97 → 97 |

- ＊ +0.910 是負的格變得沒那麼負：例如 N16 s1、128 GiB、nfs、閒，O-PF 的上限 −1.735 → −0.826，因為 oracle 的 GPU 規則改用 APC 後變快，對手不變。前 8 大都是這一型，都還是負的。
- **SemiAnalysis 的 O-GPU 按 N 分**（≥15% 格數，3 個 seed 加總）：N16 34 → 28、N32 80 → 77、N64 109 → 107、N72 110 → 109。
  - 單一變體的最大值：N16、N32 下降（例如 N16 s1 0.4617 → 0.4360）；N64、N72 上升（例如 N72 s3 0.5294 → 0.5682）。上升是因為 BELP 進了 ④ 的 oracle。
- Mooncake、Bailian 加了 APC 幾乎不變：最大的變化 0.010，≥15% 的格數只有 Mooncake O-GPU 多 2 格。
- 寫入飽和照算，另報：只有 Mooncake 有。O-ORD、O-PF 各 54＋54 格，O-GPU 15＋18，O-DROP、O-NOW 各 3＋3，O-ALL(report) 2＋2。
- TRX40 的 9 個變體（TraceLab c1／c2／c4、FreeInference 6 週）在 `origin/icpe-next-trx`，由 icpe-next-trx 自己報。

**接下來**
- W5／排隊鏈在 `CHAIN_APC_OK` 之後自動開始，12:08 起跑 W5 Bailian coder，接著是 Mooncake 2 個，再來是排隊（SemiAnalysis 12 個）。
- W7 照使用者的指示停著。

---

## 2026-11-07T10:15+08:00 — W7（合成參數地圖）：使用者決定、Rust 生成器完成並驗證、PREREG 偏離 12 登記並 push

**要證明什麼**：NEXT_PLAN W7，在哪些工作負載條件下預測的上限 ≥ 15%。只給 CCGrid 版，ICPE 論文不用。

**使用者的決定**
- 約 09:35，AskUserQuestion：9 維全部；LHS 10d × 3 seed（270 份）；逐點判＋最多加密一輪；細節交給 icpe-rux、push 前給使用者看。
- 之後補充：「需要用 rust 寫」。
- 10:05 之後看過草稿：「push 並通知」；GPU 選「固定 111.0625 GiB」。

**做了什麼**（`code/icpe/rux_x1fast/w7/`，設計見 `rux/W7_DESIGN.md`）
- `w7gen`：Rust，無外部 crate，負責 LHS 設計和合成紀錄。輸出的原始陣列由 `w7_pack.py` 先做獨立檢查，再用既有的 `x1_nb.write_nb` 包成 nb。
- 引擎不改，用 v8 48e4efce。
- 驗證全部通過：
  - 格式、鏈式前綴、時刻遞推、固定並行、鍵的來源、決定性、LHS 分層；
  - WS_ref 用 Rust 和 Python 兩份實作，逐位元相同；
  - 實現值和設定值相符；
  - 端到端煙霧測試加中斷後接續。
- 設計中途的修正：上下文上限原本草稿用 128K，會把長上下文點的輪數截到不到 1/10。真實的 SemiAnalysis 最大到 990,208 token，所以改成 1M。這是在讀任何結果之前改的。
- 真實紀錄的活工作集 WS_ref（run `20261107-092554-icpe-rux-w7-ws-real-calib`）：Bailian coder 361、SemiAnalysis N16 s1 439、N72 s1 870 GiB。CPU 水準取 WS_ref × 2^−6..2^2，涵蓋這個範圍。
- 試跑（只量時間，上限表沒讀）：最重的一份 1,432 s、中位數大小的 107 s，都是 8 個 worker、和正式 APC 鏈同時跑。正式跑 270 份估計約 7–11 小時〔推算〕。
- 揭露：看進度時讀到 5 行 sweep 進度，每行有一個 job 的 mean_ret（W7_DESIGN §5）。
- PREREG 偏離 12：只在檔尾 append，85 行、0 行刪除。
- **號碼的更正**：
  - h200 09:10 寫的「h200 下一個偏離編號是 12」，我讀成「給 icpe-rux 用 12」。h200 10:2x 回覆：原意是 h200 自己下一個要用 12。
  - h200 核過沒有撞號，照收；origin/icpe-next-a 已同步（0a9156c）。h200 自己下一個從 13 開始。
  - 所以偏離 12 標題的「號碼由 `h200` 給」實際是 h200 事後同意，不是事先給的。PREREG 只能 append，這一點記在這裡。
  - 之後的號碼會先明確向 h200 要。

**接下來**：正式 APC 鏈 → W5／排隊鏈。
- **W7 先停**（使用者 11/7 約 10:25：「W7 先停」）：
  - 排在 W5／排隊鏈後面的 W7 正式跑已取消，當時還在等、沒有開始。沒有任何 w7-formal 的 run 目錄，也沒有 W7 的 sweep。
  - 偏離 12 的登記與程式（294d2bd7）保留。要再開時用 `run_w7.sh 16`。

## 2026-11-07T09:20+08:00 — 偏離 10 的正式 APC 重算開跑（第 1 個變體通過）；偏離 11（W5＋排隊）登記並 push（4a713386）；W5、排隊的正式腳本完成，W5 在 SemiAnalysis N16 s1 跑完

**要證明什麼**
- 偏離 10：每個紀錄變體的主結果格，加 APC、BELP 之後的上限表（版本 C）。前提是只用版本 B 的 16 個規則時，能逐值重現版本 B。
- 偏離 11：W5 和排隊兩個敏感度，各自和版本 C 的同一格比較。

**使用者的決定**
- 偏離 10（APC）：全部登記變體的主結果格；加 BELP；版本 C 成為正式主表；push 並通知（commit 935dcd75）。
- 偏離 11：AskUserQuestion，11/7 09:01:38 之後、09:03:39 之前。
  - W5：7 個代表變體 × 111.0625／128／256；敏感度，每組事後挑較快格式。
  - 排隊：全部格，重負載只報告；敏感度，主結果標「不含排隊」。
  - push 並通知。
- 號碼由 h200 給（11）。h200 已回覆：核過，只在檔尾加一節、0 行刪除；origin/icpe-next-a 已同步 PREREG.md；下一號是 12。
- 已通知 icpe-main（四個決定的摘要）。

**正式 APC 重算**（RUX3DPRO 這邊）
- 順序：SemiAnalysis 12 個 → Mooncake conversation、toolagent（R1 R2）→ 轉 Bailian thinking／traceA → Bailian coder／thinking／traceA。
- 腳本：`run_apc_formal.sh`。
- 引擎：v8 `main.rs` 48e4efce，sweep code_sha `3280b1c8799b+rse7f6ee8f`，`py_fallback_jobs=0`。
- 第 1 個變體 SemiAnalysis N16 s1：run `20261107-090138-icpe-rux-apc-formal-semianalysis_N16_s1`。
  - 14,184 個 job，sweep 474.7 s，牆鐘 7:55，最大 RSS 1.16 GiB。
  - 重現檢查：版本 B 表的 324 個 (格, 元件) 全部對上，7 欄都是 0 個不同 → `EXPLORE_RESULT REPRODUCED`。
  - 版本 C（54 格，cake）：

| 元件 | 最大 | 中位數 | ≥15% 格數 | ≥5% 格數 |
|:--|--:|--:|--:|--:|
| O-GPU | 0.4360 | 0.0513 | 12 | 27 |
| O-ORD | 0.1818 | 0.0117 | 2 | 10 |
| O-PF | 0.0397 | −0.0054 | 0 | 0 |
| O-DROP | 0.0258 | 0.0050 | 0 | 0 |
| O-NOW | 0.0019 | −0.0002 | 0 | 0 |
| O-ALL(report) | 0.2300 | 0.0136 | 5 | 10 |

  - 收在 `results/icpe_next/apc/semianalysis_N16_s1/`。
  - ⚠ 這個 run 有存逐請求 npz（`req/`，11 GB，沒進 git）。原因：它開始時 bash 讀的是舊版腳本，舊版預設會存 npz。第 2 個變體起用 7a89a9b5 的版本，預設不存。

**W5 正式腳本** `run_w5_formal.sh`（偏離 11 A）
- 每個變體：
  - 只跑 W5 那一半（`RUX_W5=1`，CPU 111.0625／128／256 × CPU 17 × GPU 18）。
  - BF16 那一半從偏離 10 的正式 sweep 篩出同一個 rule、同 3 個容量。
  - 之後依序跑：
    1. `explore_w5.py`：三份上限表都改用 `analyze_v8`。
    2. `w5_mechanism.py ... allowed`。
    3. `collect_v8.py` → `results/icpe_next/w5/<變體>/`。
- SemiAnalysis N16 s1：
  - run `20261107-091018-icpe-rux-w5-formal-semianalysis_N16_s1`：4,728 個 job，sweep 314.4 s，`py_fallback_jobs=0`；BF16 與 W5 各 5,508 列，組合相同。
  - **核對**：
    - `base` 上限表和版本 C（`apc/semianalysis_N16_s1/verC_headroom_components.csv`）同 3 個容量的 108 個 (格, 元件) 逐值相同，只有 run_id／ts 不同。
    - 所以「和版本 C 同一格比」＝和 base 比。
  - **機制表改用 allowed**：
    - 第一次跑的 `w5_mechanism.py` 用 forced（每個組合都用 W5，是探索版的預設）。偏離 11 登記的是 allowed（每組事後挑較快格式），所以改了程式：第 3 個參數選 forced 或 allowed，並加 `fmt_allowed` 欄。
    - 用 run `20261107-091642-icpe-rux-w5-formal-mech-semianalysis_N16_s1` 重做（只讀上一個 run 的輸出），重新 collect，MANIFEST 列了兩個 run。
    - 這個變體 W5 較快的組合是 5,507／5,508，所以兩種算法的結果相同。
  - **結果**（cake，18 格；只報告，主結果不變）：

| 元件 | 版本 C 最大 | allowed 最大 | 版本 C ≥15% | allowed ≥15% | 差的中位數（百分點） |
|:--|--:|--:|--:|--:|--:|
| O-GPU | 0.1867 | 0.0087 | 2 | 0 | −2.36 |
| O-ORD | 0.1818 | 0.0528 | 2 | 0 | −2.19 |
| O-PF | 0.0015 | 0.0082 | 0 | 0 | −1.95 |
| O-DROP | 0.0258 | 0.0193 | 0 | 0 | −0.20 |
| O-NOW | 0.0007 | 0.0013 | 0 | 0 | +0.13 |

  - NP:53：升高超過 5 個百分點的有 3 個 (格, 元件)，全部在 nfs：
    - O-PF 2 格，中位數 +27.8 百分點。④ 和 ② 的慢層載入各少 95%、84%。
    - O-ALL(report) 1 格。
    - 機制和探索時相同：INT4 讓 CPU 層裝得下更多，慢層載入大減。詳細見 `w5_mechanism_rows.csv`。

**排隊正式腳本** `run_queue_formal.sh`（偏離 11 B）
- `RUX_QPOST=1`，跑全部 9 個容量 × CPU 17 × GPU 18。
- 之後跑 `explore_queue.py`：
  - 和偏離 10 的正式 sweep 做端到端比對：只有 TTFT 衍生的欄位可以不同。這次加上「組合數必須全部對上」的 assert。
  - 用版本 C 的表做對照，並按 (LRU, LRU) 的平均等待分級：輕 <2 s、中 2–20 s、重 ≥20 s，重的只報告。
- 端到端不通過就停。之後 collect → `results/icpe_next/queue/<變體>/`。
- 還沒跑。

**接下來（自動接續）**
- 正式 APC 鏈（Bailian 為止）跑完並印 `CHAIN_APC_OK` 之後，接著跑：
  - W5：Bailian coder、Mooncake conversation R1、Mooncake toolagent R1。
  - 排隊：SemiAnalysis 12 個變體。
- 任何一步失敗就停。
- TraceLab c1、FreeInference 第 1 週的 W5 交給 icpe-next-trx（通知中）。
- 之後是 W7：先把設計和要決定的點給使用者，不在第 2–5 項的授權範圍內。

**卡住什麼**：無。

**需要使用者決定什麼**：目前沒有。W7 開始前會問。

## 2026-11-07T08:40+08:00 — PREREG 偏離 9（W3 線索 (a) 的規則）登記並 push（8f04a91c），之後補「偏離 9 更正」（e7e0c778）

- **使用者的決定**：
  - 使用者在 icpe-main 決定（acvlab 約 08:28–08:30），icpe-main 轉達：W3 要在 TRX40 跑之前登記，由 icpe-rux 寫。
  - 使用者在這裡確認（AskUserQuestion，08:29:19 之後、08:35:26 之前）：比例「相對 ②（PR:21）」、門檻「f > 0.5」、彙總「逐格，任一格成立就算」、「由 icpe-rux 直接 append」。
  - **更正**：偏離 9 原本寫「約 08:45」「約 08:50」，都錯了。已用「偏離 9 更正」（commit e7e0c778，使用者同意後 push）改正，同時照實補上方向：逐格任一成立的彙總偏向推翻。
- **號碼**：h200 給 9。origin 上各分支都只到偏離 8。
- **內容**：只登記 W3，追加在 PREREG 檔尾，不改任何舊行。
  - 線索定義、cue 的產生與對齊檢查；
  - 紀錄（TraceLab c1／c2／c4、FreeInference 6 週）與格；
  - ③、f_c 的算法；判定規則；方向；看過的結果。
- **程式**：`analyze_w3.py` 照登記輸出 `w3_verdict.csv`。
  - ③ 要 20 個組合齊全；
  - 只判 5 個元件；
  - 逐格判定；
  - 另報寫入飽和的格數。
- **TRX40**：W3 的 sweep 與分析等登記 commit 進 git 才開始。cue 已對齊。

---

## 2026-11-07T08:35+08:00 — 排隊：Q3-a 由我決定、一階近似完成並在 SemiAnalysis 上跑；TraceLab、FreeInference 交給 icpe-next-trx；結果改放 `results/icpe_next/v8_explore/`

**使用者的指示（原話）**
- 「Q3-a（排隊） 我不知道你決定。 開始跑 跟要不要通知其他 session 通知icpe main 就好」
- 「icpe-next-trx 目前空的 叫他模擬 跟 CacheWise 都先不跑」「說錯是把 TraceLab 在 給他」「FreeInference 也是」「分工一下」
- 「你需要說一下資料計算完放那github?」

**排隊**
- **Q3-a 的決定**（SEMANTICS §2b）：④ 只知道紀錄的順序。快取決定照紀錄順序做，排隊在事後套上：工作守恆的 closed-loop FCFS，`qpost=1`。
- **驗證**：
  - SemiAnalysis 上 48／48 通過：到達拉開時＝沒排隊；Python 精確重算逐位元相同；快取狀態不變。
  - 重排程度：輕負載小（位移中位數 18 名），重負載大（約 2,000 名）。
- **結果**（V8_FEATURES §4c；SemiAnalysis N16／N32／N64／N72 s1 × CPU 8／111／1,024）：
  - 輕負載上限幾乎不變。
  - 負載愈重上限愈大：O-GPU ≥15% 的格在 N64、N72 變成 18／18。這和 NP:266「不排隊對預測有利」的方向相反。
  - 重負載的格不可靠。
- **端到端**：只有 TTFT 衍生的欄位不同。第一次檢查把 NaN 判成不同，是檢查程式的錯，已修。

**分工**
- **icpe-next-trx**：TraceLab c1／c2／c4、FreeInference 6 週的重現檢查 → W3、APC／BELP、W5。使用者在 TRX 那邊也確認了，已開始。
- **RUX3DPRO**：SemiAnalysis（排隊、APC、W5）、Mooncake、Bailian。
- **CacheWise**：先不跑。
- **通知**：只通知 icpe-main（v6／v7／v8 摘要）。

**結果放法**（`results/icpe_next/v8_explore/README.md`）
- 各 session 放自己的分支，路徑 `results/icpe_next/v8_explore/<實驗>/<變體>/`。
- 內容：sweep 的 `x1_sweep.csv.xz`、分析的 CSV、`MANIFEST.json`（run、commit、引擎與檔案的 sha256）。
- 不放：原始資料、nb 檔、cue 檔、逐請求 npz。
- 已收：APC、W5（Mooncake toolagent、SemiAnalysis N16 s1、Bailian coder）、排隊（SemiAnalysis 4 個變體）。

---

## 2026-11-07T04:30+08:00 — 第 4 段（v8：第 2–5 項的新功能，只有 Rust）：APC、W5 完成並接進 sweep；W3 引擎完成、缺資料；排隊**要使用者決定 Q3-a**

報告：`next/rux/V8_FEATURES.md`；語意：`next/rux/SEMANTICS_2to5.md`；程式：`code/icpe/rux_x1fast/v8_features/`（最終版 `main.rs` sha256 `d1502d2d…`；獨立審查後改了 3 處，見 V8_FEATURES §3b）。

**驗證（沒有 Python 參考，照使用者同意的做法）**
- **新功能關閉時**：v8＝Python。3 份紀錄 × 4 組子集，CSV 177,072 格、npz 11,424 個陣列，0 個不同（`…-v8g-reg-mc／bl／sa`）。G1、xc_sa 0 個不同（`…-v8g-off-g1`、`…-v8g-off-xc_sa`）。
- **四項新功能**：最終版 `smoke_v8.py` 在 3 份紀錄上共 2,088 項，**0 個 FAIL**（`…-v8g-smoke-mc／sa／bl`）。檢查方式：
  - 退化檢查：新功能設成等於沒開時＝v7。
  - 精確重算：排隊。
  - 不變量：W5 的位元組。
  - 稽核：APC 每 200 個請求從頭重算 GPU 引用數。
  - 兩份實作互比：BELP 的捷徑＝每塊全掃。
- **端到端**：
  - sweep 管線重現版本 B 的 toolagent 上限表（306 列逐值相同）。
  - APC 一次逐出整個 session 時＝LRU／BEL。
  - W5 走新路徑但參數等於沒開時＝BF16：17 × 16 個規則，254,592 格相同。
  - BELP 捷徑：832 格相同、快 2.3 倍。

**探索（Mooncake toolagent R1／R2；不是正式結果）**
- APC 加進 ② 和 ④（另外跑了 SemiAnalysis N16 s1、Bailian coder；三份都重現版本 B 的表，逐值相同）：
  - Mooncake toolagent、Bailian：上限最多變 0.03 個百分點。
  - **SemiAnalysis（長 session）明顯**：APC 在每格都是 ② 的最佳，② 最多快 28%；O-GPU（④ 加 BELP）上限中位數 8.5% → 4.6%，111 GiB、nfs 從 18–21% 降到 6–9%，≥15% 的格 6 → 4。
  - **④ 一定要有 BELP**：沒有的話，SemiAnalysis 的 O-GPU 最大上限會被壓到 25%（有 BELP 是 44%）。
  - 原本的 `next_sim_analyze.py` 會把 BELP 當成不預測。所以另寫 `v8_features/analyze_v8.py`（原檔不改）：BELP 只進 ④ 的 O-GPU。已檢查和原本的分析、獨立算法都 0 個不同。
  - 建議 PR:95 的重算至少包含長 session 的紀錄（SemiAnalysis、TraceLab、CacheWise）。
- W5（CPU 111／128／256 GiB；另外跑了 SemiAnalysis、Bailian：多半照 NP:53 下降，SemiAnalysis 的 O-GPU ≥15% 的格 4 → 0、Bailian 8 → 2；下面的上升主要是 Mooncake toolagent 的現象）：
  - 上限有升有降：O-GPU −6 到 −8 個百分點；O-DROP、O-ORD 上升，nfs 的格最多 +39 個百分點。
  - **違反 NP:53 事先登記的預期**（「下降或不變；若上升，要查原因」）。
  - 查了上升超過 5 個百分點的 45 個 (格, 元件)：O-DROP、O-ORD、O-PF 的 ④ 用 INT4 多出來的 CPU 容量，慢層載入減 79–99.5%，② 只減 13–26%；大多在 nfs。

**獨立審查**（另一個 agent 只讀程式、對照 SEMANTICS）
- 沒有找到引用數、所有權或選擇邏輯上的錯。
- 改了 3 處：
  - F2：W5 不重複計寫入；
  - F6：放寬預取進 GPU 的斷言；
  - F7：稽核多重算 ref_live、ref_cpu、cpu_used。
- 改完重跑全部：2,088 PASS，對 Python 0 個不同。W5 探索的上限逐列不變。
- 另外 4 點和 v7 的 wt 行為一致，所以不改、寫明（APC × CPU 預取、CPU 丟 session 後的 GPU 前段等）。

**還沒做／要決定**
- **Q3-a（使用者）**：排隊的 ④ 知不知道排隊造成的新服務順序。
  - open-loop（`q=1`）在 SemiAnalysis 上平均要等 2,000–19,000 秒；照紀錄順序的 closed-loop（`q=2`）多等 30–300 倍。兩個都不能用。
  - 正確的工作守恆 closed-loop 要先決定這件事。
- **W3（icpe-data 在 acvlab 上跑）**：工具鏈都好了，指令在 `v8_features/README.md`。流程是 make_cue → check → sweep CUEP／CUEO → analyze_w3，③ 只要約 450 個 job，不用改 nb 檔。這台記憶體不夠跑 TraceLab，所以只在 SemiAnalysis 上用代用 cue 跑通管線。TraceLab 原始紀錄上，「輸出有工具呼叫」和「下一輪由工具結果觸發」一致 96.9%。
- **登記（h200／icpe-data）**：SEMANTICS_2to5.md 是 PREREG 的草稿。
- **審查（icpe-sim 或另一個 session）**：審語意與程式的對應，決定要不要併進 `code/p3x/x1fast`。

**規則遵守**
- 每個 run 都用 `rux_run.sh`。
- 舊執行檔上的兩個測試 run（`20261107-033505-…-v8-smoke-all-sa／bl`）被最終版取代，是我停掉的，所以沒有 exit_code。停掉前 SemiAnalysis 476 PASS、Bailian 593 PASS，0 FAIL。
- 沒有跑正式 block、沒有動佇列、沒有改別人的檔。

---

## 2026-11-07T03:45+08:00 — 第 2 段（v6：慢層上限的 Rust 版）與第 3 段（v7：加速）完成，**和 Python 逐位元相同**；第 4 段（第 2–5 項的新功能，只有 Rust）進行中

**使用者的指示（原話）**
- 「ok push 跟 後段都你做不需要跟其他session 做」
- 「好就這樣」：第 2 段之後接著做低風險的加速。
- 「也就是 python 不寫完整移植 rust」：之後的新功能只寫 Rust。
- 「第 2 到 5 細節 都可以你決定 符合實驗命題且有規範 其他 paper 合理這樣 我要睡了 都交給妳」
- 「這太電腦改用更高執行緒加速 也可」：Rust 驗證改用 16 個 worker。

**第 2 段：v6（`code/icpe/rux_x1fast/v6_slowcap/`；報告 `next/rux/V6_SLOWCAP.md`、`V6_rows.csv`）**
- 慢層容量上限照 `x1_sim.py` 一字不改移植。`code/p3x/*` 沒動。
- **功能打開**：3 份紀錄（Mooncake conversation R1、Bailian coder、SemiAnalysis N16 s1）× 4 組子集（main、wt、pfgpu、noise B1／B2）× 7 種慢層規則。對 Python：CSV 177,072 格、npz 11,424 個陣列，**0 個不同**。
- **功能關閉**：G1 128,064 格、xc_sa 1,280 格，0 個不同。
- **第一版有錯，已改**：`make_room` 尾端少了一次 `_slow_enforce`（py:1580–1581）。Mooncake 主子集有 225 格不同，照規則停下、改好、全部重跑（run `20261107-004914-…-v6-val-mc`，§3c）。
- **速度**：
  - 主子集 12 worker，Python ÷ Rust＝26.3×（Mooncake）、51.1×（Bailian）、44.2×（SemiAnalysis）。
  - 單執行緒同一個 job 20–66×。
  - 正式格子每種紀錄在這台約 0.5–2.3 小時〔推算〕。

**第 3 段：v7（`v7_fast/`；報告 `V7_FAST.md`、`V7_rows.csv`）**
- 做了 ANALYSIS §3d 的候選 2（LRU2 有序集合）、3（PENS 成本快取）、5（快速路徑＋補記 `clean_q`）。
- 驗證模式與一般模式各對拍 3 份紀錄 × 4 組子集，重用 v6 那次的 Python 參考，**0 個不同**。G1、xc_sa 也 0 個不同。
- 單執行緒 v6 ÷ v7：
  - Bailian 慢層 LRU2 12.6×、PENS 2.2×；
  - Mooncake LRU2 3.1×；
  - SemiAnalysis 各規則 1.16–1.26×。
- 16 worker 主子集：Bailian 54.5 → 37.1 s，SemiAnalysis 71.8 → 60.2 s。
- 候選 1（慢層 BEL lazy heap）、4（7 種規則共用前段）**沒做**。

**第 4 段（進行中）：第 2–5 項（`v8_features/`；語意 `next/rux/SEMANTICS_2to5.md`）**
- 語意由 icpe-rux 決定（使用者授權），依據 PR、NP、SS 與外部系統。
  - APC：session LRU，從尾端一塊一塊逐出。wb 下第一次逐出時整個 session 寫進 CPU；④ 也有 BELP。
  - 排隊：FCFS-1，只跑固定並行的紀錄。
  - W3：CUEP（釘住）／CUEO（立刻逐出）。
  - W5：CPU 層 INT4＋M3-g 的一次拆完殘差。
- **要登記**：PR:57、95 要求跑之前登記。PREREG 由 h200／icpe-data 寫，`SEMANTICS_2to5.md` 是登記內容的草稿。
- **W3 的資料**：nb 檔現在只存「使用者觸發」，「none」會被當成工具觸發。要 icpe-data 在 nb 檔多存一欄「工具觸發」。
- 狀態與驗證結果見下一節（做完再寫）。

**規則遵守**
- 每個 run 都用 `rux_run.sh`（cmd.sh、context.txt、exit_code、mem.log）。v6／v7 共 21 個 run，MemAvailable 最低 10.4 GiB。
- 沒有跑正式 block、沒有動佇列、沒有改別人的檔。CacheWise 不在這台。

---

## 2026-11-07T00:40+08:00 — 第 1 段（分析）完成：未支援的 7 項模擬的形態、時間、Rust 缺口、加速候選；**停下來等使用者決定**

完整報告：`docs/icpe_brief_20261030/next/rux/ANALYSIS.md`；摘要 CSV：同目錄 `ANALYSIS_rows.csv`（每列有 run_id、ts、host、threads、load）。

**做完什麼**
- 前提：重新抽查第一份 prompt 的對拍，全部 0 個不同；雜訊 B1／B2 的比對上一段沒存檔，重跑一次（run `20261107-001138-icpe-rux-recheck-noise`）。上一段有 7 個 run 目錄缺 `exit_code`（比對 run 都有），之後的 run 一律用 `code/icpe/rux_x1fast/stage1/rux_run.sh`（寫 exit_code、每 5 s 記 MemAvailable 與 load、< 3 GiB 自動停）。
- pull：paper repo 的遠端各分支沒有新 commit；kvsweep fast-forward 到 `6d3c5f9`。
- Bailian coder 在這台轉成 npz，指紋和 acvlab 相同（`192949471c…`）。
- 慢層上限（項目 1）在 Mooncake、Bailian、SemiAnalysis 各一個代表 job 上量 Python（有／沒有上限、7 種慢層規則）與 Rust v5（沒有上限），加 cProfile 與操作次數；用 `jobs_for` 數 job；量第一次超過上限的位置 r\*。
- 其他項目（APC、排隊、W3、W5）量了最接近的 v5 job 形狀。

**主要發現**
- Python ÷ Rust v5（沒有上限、逐請求指紋全部相同）：這台 17–50 倍（Mooncake、Bailian 約 20 倍，SemiAnalysis 約 44–50 倍）。
- 慢層上限讓 Python 變慢的原因有兩種：
  - SemiAnalysis：7 種規則都約 1.9×，因為開上限時關掉 `_fix` 快速路徑（`_fix` 呼叫 ×3.52）。
  - Mooncake／Bailian：LRU2、PENS、BEL 每次丟 session 都掃全部慢層 session → 7.8–125×（Bailian BEL 一個 job 76 s，1.29 億次 `next_req`）。
- Rust 照抄版〔推算〕：比 Python 快 20 倍以上，但 Mooncake／Bailian 的 LRU2、PENS、BEL 仍會是 v5 一般 job 的 30–100 倍 → 之後值得做 lazy heap 等加速（逐位元相同）。
- 第一次超過上限在 Mooncake／Bailian 約第 45–57% 個請求、SemiAnalysis 約 90%：在那之前 7 種慢層規則的狀態相同，可共用。
- ARC 與 S3FIFO 在 3 份紀錄的代表 job 輸出逐位元相同（SemiAnalysis 上 BEL 和 LRU 也相同）；只看了 1 個 job，報告裡有說明。
- Rust 版最大的風險：`clean_q` 的順序取決於每一次 `_fix` 呼叫，v5 的三個快速路徑（rs:1198、1302、1366）開上限時必須關掉。

**卡住什麼**：無；照指示停下來等決定。

**需要使用者決定什麼**（細節見 ANALYSIS.md §9）
- D0：項目 2–5 的 Python 參考實作由誰寫。
- D1-1：「被丟的對話回來時全部重算」照 Python（別人還持有的共用前綴照常還原）是否就是 PREREG 的意思。
- D1-2：「先丟乾淨副本」「×7 配對」要不要請 h200 寫進 PREREG。
- D1-3：第 2 段只做照抄版（建議），還是接著做低風險的加速。
- D1-4：慢層上限要跑的格子（不卡第 2 段）。
- D1-5：第 2 段驗證也涵蓋 wt 與預取進 GPU（我的預設：要）。
- D1-6：TraceLab 要不要在這台量（記憶體 22 GiB）。
- 項目 2–5 的語意選項。

**需要別的 session 做什麼**：無（第 2 段做完才通知 icpe-sim）。

---

## 2026-11-07T00:15+08:00 — 環境建好；**對拍全部通過**：npz 指紋 13／13 SAME、G0（py／rust）、xc_sa（py／rust）、雜訊 B1／B2 全部 0 個不同

**機器（實測，開跑前記錄）**
- `nproc`＝**16**；Ryzen 7 7800X3D（8 核 16 執行緒、1 socket）。
- `free -g`：total **22**、used 2、available 20；swap **6 GiB**（開跑前已用 0 MiB，以此為基準）。
- `df -h /`＝`/dev/sdd` **1007 G**，可用 947 G；`$HOME=/root` 就在 `/dev/sdd` 上（`stat -c %d`＝2096，與 `/` 同）。
- `/etc/wsl.conf`：只有 `[boot] systemd=true`。Windows 側 `C:\Users\asdf9\` **沒有 `.wslconfig`**（使用者未提供內容）。
- `MemTotal` 24,028,820 kB＝22.9 GiB；`MemAvailable` 開跑前 21,584,640 kB。
- 沒有 NVIDIA 顯卡（`nvidia-smi` 不存在，只有 WSL 的 `/dev/dxg`）。

**0. 目錄**
- `$ROOT=/root/icpe-next`，在 `/dev/sdd` 上。底下 `venv/ uv-python/ datasets/ runs/ hf-home/ tools/`（另有 `x1fast-work/`）。
- 程式和資料分開：repo 在 `$HOME/paper-hierarchical-kv-state`（照 TRX40 的配置）。

**1. Repo**
- clone `https://github.com/s990093/paper-hierarchical-kv-state`（public），從 **`origin/icpe-next-data` @ `8d70cbca`** 開分支 **`icpe-next-rux`**（`git checkout -B icpe-next-rux origin/icpe-next-data`）。
- repo 的 `.claude/settings.json` 把 `HF_HOME`、`UV_*` 指到 `/ssd7`、並設 `HF_HUB_ENABLE_HF_TRANSFER=1`；每條指令都明確覆蓋或 unset（`tools/env-rux.sh`）。

**2. venv（`$ROOT/venv`）**
- Python **3.12.15**（`cpython-3.12.15-linux-x86_64-gnu`，`uv python install --install-dir $ROOT/uv-python --no-bin`）。
- `uv pip freeze` **正好 5 個**：`numpy==2.3.5`、`pandas==3.0.5`、`pyarrow==25.0.1`、`python-dateutil==4.0.0`、`tzdata==2026.6`。
- 只用來建 Python／下載的 `uv 0.13.7`、`hf 2.4.0`（huggingface_hub）放在 **另一個** venv `$ROOT/tools/hf-venv`，不污染模擬用的 venv。

**3. Rust（`$ROOT/tools`）**
- rustup 裝到 `RUSTUP_HOME=$ROOT/tools/rustup`、`CARGO_HOME=$ROOT/tools/cargo-home`，`--default-toolchain 1.99.1`：
  - `rustc 1.99.1 (58cf8f9c3 2026-10-13)`、`cargo 1.99.1 (5f94df478 2026-08-27)`——和 acvlab／TRX40 相同。
- 環境：`X1_CARGO=$ROOT/tools/cargo-home/bin/cargo CARGO_HOME=$ROOT/tools/cargo-home RUSTUP_HOME=$ROOT/tools/rustup X1_RUST_WORK=$ROOT/x1fast-work`。
- 編譯照 `code/p3x/x1_rust.py`（`--profile fast --offline --locked`，不加 `target-cpu`）：試編 run `20261106-233220-icpe-rux-buildtest` exit 0、**3.79 s**。
- `code/p3x/x1fast` 的 `[dependencies]` 是空的，所以 `--offline` 不需要 registry 快取。

**4. 資料**（run `20261106-233235-icpe-rux-dl`，exit 0；`setup_3090_data.sh mooncake semianalysis bailian tracelab`）
- 環境照 TRX40：`TIARA_SKIP_VENV=1 TIARA_HOME=$ROOT TIARA_DATASETS=$ROOT/datasets HF_HOME=$ROOT/hf-home TIARA_PY=$ROOT/venv/bin/python TIARA_HF_CLI=$ROOT/tools/hf-venv/bin/hf`。
- **CacheWise 沒有下載、沒有出現在這台**（`ls $ROOT/datasets` 只有 mooncake／semianalysis_cc_traces／qwen-bailian-usagetraces-anon／tracelab）。

| 資料 | 要求的值 | 本機實測 | 結果 |
|:--|:--|:--|:--|
| Mooncake conversation | `b8cbb061…` | `b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df` | SAME |
| Mooncake toolagent | `48a2db1a…` | `48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71` | SAME |
| Mooncake commit | `d750cdda` | 6 個檔 git-blob OK（GitHub tree API） | SAME |
| SemiAnalysis revision | `23f152f6` | `23f152f6f0f9399a85901b89a6458def0ef16729` | SAME |
| SemiAnalysis `traces.jsonl` | `29b6a19e…` | `29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276` | SAME |
| TraceLab release | v0.0.2 / `11ce51ec…` | `11ce51ec0a25e3d1d95b025bca2f7d1647e47571eb7cc968acd5fc64d4b4fb65` | SAME（完整值相符） |
| Bailian commit | `5f7439c5` | `5f7439c51ec248a0c585f7d90a41a6f57773b912`；4 個 jsonl 的 sha256＝LFS pointer oid | SAME |

- Mooncake 單位驗算（規則 6，block 512）：conversation 12031／12031、synthetic 3993／3993、toolagent 23608／23608、mooncake 23608／23608，**rate 全部 1.000000**。
- 下載後的單位驗算與 `m8_trace_oracle.load_trace` 斷言全部通過（`[mooncake] OK`、`bailian OK`、`tracelab OK`）。

**轉換**（run `20261106-233531-icpe-rux-convert-sa`，exit 0）
- `next_b_convert.py semianalysis --N 16 32 64 72 --seeds 1 2 3 --out-dir $ROOT/datasets/x1_nb --p3x $REPO/code/p3x`，再跑 `--N 16 --seeds 1 --trunc-128k`。共 13 個 npz。
- 規則 6：`len(hash_ids)×64=in` **98827／98827 = 1.0000**。每個變體 `n_req=98827 n_sess=2090 chunks=42212088`；截斷版 `n_req=40799 n_sess=2079 chunks=4785716`。
- 用 **HEAD 的 `code/p3x`**（`x1_nb.py` sha256 `8251e258…`，和 STATUS_DATA 11/6 09:35 宣告的 acvlab 值 `8251e25` 相同）。TRX40 當時用的是 `git archive 6b17191`；兩版 `x1_nb.py` 的差異只在 `check()` 向量化，且 `write_nb` 是 `st, _ = check(...)`——稠密 id 被丟棄，**輸出陣列不受影響**；下面的指紋也證實了這點。

**npz 內容指紋（不含 `meta_json`）**
- run `20261106-233907-icpe-rux-npz-digest-nometa2`（算）、`20261106-233920-icpe-rux-compare-npz-digest`（比）。
- 比對基準：`results/icpe_next/b/traces/npz_digest.csv`（acvlab，run `20261105-090851-icpe-data-npz-digest-nometa`）。
- **`DIGEST_RESULT IDENTICAL n=13 bad=0`**——13／13 SAME。

**5. 對拍（全部 0 個不同）**

| 項目 | run_id | 結果 | sweep | 牆鐘 | CPU(user) | 最大 RSS | swap |
|:--|:--|:--|--:|--:|--:|--:|:--|
| (a) G0 Python | `20261106-233928-icpe-rux-g0-py` | 4,416 列 × 29 欄＝128,064 格，**0 不同** | 922.9 s | 924.5 s | 10,788.3 s | 386 MiB | 0 |
| (b) G0 Rust | `20261106-235852-icpe-rux-g0-rust` | 同上，**0 不同** | 40.3 s | 46.1 s | 429.0 s | 308 MiB | 0 |
| (c1) xc_sa Python | `20261107-000317-icpe-rux-xcheck-xc_sa-py` | 32 列 × 40 欄＝1,280 格，**0 不同** | 46.4 s | 54.5 s | 448.2 s | 2,111 MiB | 0 |
| (c2) xc_sa Rust | `20261107-000429-icpe-rux-xcheck-xc_sa-rust` | 同上，**0 不同** | 2.07 s | 11.7 s | 19.9 s | 2,855 MiB | 0 |
| (d1) 雜訊 B1 | `20261107-000520-icpe-rux-p4-noise-subset2/sa_p4_b1_semianalysis_N16_s1_P_B1` | 1,394 列 × 29 欄＝40,426 格，**0 不同** | 19.4 s | 20.3 s | 226.6 s | 1,029 MiB | 0 |
| (d2) 雜訊 B2 | `20261107-000520-icpe-rux-p4-noise-subset2/sa_p4_b2_semianalysis_N16_s1_P_B2` | 1,360 列 × 29 欄＝39,440 格，**0 不同** | 17.2 s | 17.8 s | 203.8 s | 1,029 MiB | 0 |

- 全部 `--workers 12`（這台 22 GiB，照指示 ≤12）；`nice -n 10`。全部 `Swaps: 0`。
- 記憶體監看（15 s 一次，`MemAvailable < 3 GiB` 就停整組）：G0-Python 最低 **19.93 GiB**、G0-Rust 最低 **19.96 GiB**、xc_sa-Python 最低 **18.08 GiB**；xc_sa-Rust 與兩個雜訊 run 沒開監看 → 那幾項的 MemAvailable 與 load 峰值 **NOT_MEASURED**（`/usr/bin/time -v` 的 `Swaps: 0` 有量到）。load 峰值：G0-Python 12.24、G0-Rust 6.41、xc_sa-Python 5.79，其餘 NOT_MEASURED。
- **(a)(b) 比對**：`next_b_g0.py results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz <run>/out/x1/x1_sweep.csv`（run `20261106-235845-icpe-rux-compare-g0-py`、`20261107-000309-icpe-rux-compare-g0-rust`）。
- **(c) 比對**：`next_b_g0.py results/icpe_next/b/xcheck/xc_sa/x1_sweep.csv.xz <run>/out/x1/x1_sweep.csv`（run `20261107-000429-icpe-rux-compare-xc_sa-py`、`20261107-000452-icpe-rux-compare-xc_sa-rust`）。xc_sa 的參數照 queue 的 `xcheck` 定義：`--cpu 64 --policies LRU GDSF O-BEL O-PF --slows ssd --modes cake --gpu-gib 111.0625 --gpu-write wb --gpu-policies LRU BEL TTLA:10 TTLB:10 --pf-target cpu`。**沒有 collect。**
- **(d) 做法**：`sa_p4_b1`（B1）與 `sa_p4_b2`（B2）各挑一個 block－`semianalysis_N16_s1_P_B1`、`semianalysis_N16_s1_P_B2`。每個 block 共 287 個 job，用 `--jobs-from` 取 **16 個**（等距抽）。參考檔取自 `origin/icpe-next-trx:results/icpe_next/b/<plan>/<block>/x1_sweep.csv.xz`。`--engine rust`，`py_fallback_jobs=0 {}`（B2 沒有退回 Python）。
  - ⚠ **`next_b_g0.py` 不能直接用在這兩個 block**：兩邊的 CSV 都沒有 `slow_cap_frac`／`slow_policy`，所以比對器退回 7 欄鍵，而雜訊 block 的 7 欄鍵**不唯一**（B1：1,394 列只有 33 個相異 7 欄鍵）。改用 run 目錄裡的 `cmp_noise.py`：鍵＝7 欄＋`gpu_GiB gpu_write gpu_policy pf_target noise_model noise_lambda noise_draw noise_seed`（15 欄），要求鍵唯一且每一個非鍵欄逐格字串相同。**這是這台新寫的比對腳本，不在 repo 裡。**
  - `--jobs-from` 是 **job 粒度**（一個 job 的多個 view 都會跑），所以產出列數會比 jobs.csv 的 7 元組多：16 job → B1 1,394 列、B2 1,360 列。

**速度（只代表這台的相對速度，不能拿來下任何時間的結論）**
- G0：Python 922.9 s（12 worker）→ Rust 40.3 s，**22.9×**（同一台、同 12 worker）。
- xc_sa：Python 46.4 s → Rust 2.07 s，**22.4×**。
- G0 的 worker-秒數：Python 922.9×12＝11,075、TRX40 是 687×16＝10,992（16 worker）——**每核速度與 TRX40 幾乎相同**〔推算，同程式、不同 worker 數〕。

**卡住什麼**：無。

**需要別的 session 做什麼**：無。

**需要使用者決定什麼**
- **push**：`icpe-next-rux` 目前只有本機 commit，**還沒 push**（照規則 push 前先問）。
- 之後要不要把正式的 block 分給這台跑（本節沒有跑任何正式模擬）。

**已知落差／限制**
- 這台是 **WSL2**：Windows 休眠、自動更新重開機、WSL 閒置被關都會打斷 job。今天 13:38:50 就正常關機過一次、23:00:28 才再開（本 session 前半段連不上就是這個原因）。**開跑前需要使用者去關掉 Windows 的睡眠與自動重開機**，我沒有改任何 Windows 設定。
- 記憶體 22 GiB，比 acvlab（1 TiB）與 TRX40（62 GiB）小很多；本次對拍最高單一行程 RSS 2,855 MiB（xc_sa Rust），G0 只有 386 MiB。正式 block 的 worker 數還沒量過。
- **更正（00:10 發現）**：這台**有** NVIDIA GPU。先前只看 PATH 上的 `nvidia-smi`（不存在）就判定沒有 GPU，是錯的。
  - Windows 主機：`NVIDIA GeForce RTX 3080`（DriverVersion 32.0.16.2114）＋ `AMD Radeon(TM) Graphics`（內顯）。
  - WSL 內 GPU 直通**可用**：`/usr/lib/wsl/lib/nvidia-smi` 執行成功 → `RTX 3080`、`10240 MiB`、`compute_cap 8.6`、driver `620.70.03`、CUDA UMD `13.5`、`343 W`。
  - 原本 PATH 上沒有 `nvidia-smi`；已加 `/usr/local/bin/nvidia-smi` 連結與 `PATH` 的 `/usr/lib/wsl/lib`。
  - **尚未用 PyTorch／CUDA 實測算例 → CUDA 可用性 NOT_MEASURED**（目前只到驅動層）。
  - 限制：10 GiB 顯存（acvlab 的 3090 是 24 GiB），且開機後 Windows 桌面已用掉約 1.8 GiB。sm_86（Ampere）與 3090 同一世代。
  - 對本次對拍的影響：**無**（G0／xc_sa／雜訊全部是 CPU 路徑）。
