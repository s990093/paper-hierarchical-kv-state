# ICPE 未支援的模擬：分析（icpe-rux 第 1 段，RUX3DPRO）

> 作者 icpe-rux，2026-11-07。這段**沒有寫引擎程式、沒有跑正式 job、沒有動任何佇列、沒有改別人的檔**。
> 標記：〔推算〕＝由寫出來的算式得到；〔推論〕＝讀程式或文件推出、沒量；NOT_MEASURED＝沒量。
> 每個數字的出處是 run 目錄（`/root/icpe-next/runs/<run_id>/`）或檔案:行號。摘要 CSV：`ANALYSIS_rows.csv`（同目錄，每列有 run_id、ts、host、threads、load）。

## 0. 問題是什麼／要證明什麼

- **問題**：P1–P4 之後還有 7 項模擬沒做。其中規則已登記、最先能跑的是「慢層容量上限」，但 Rust 引擎（x1fast v5）不支援，碰到就退回 Python 跑。
- **這段要回答的**（每一項都答 a–d）：
  - a. 一個 job 是什麼、要跑多少、哪些已登記、哪些要使用者先決定。
  - b. 時間花在哪（這台實測）。
  - c. 現有 Rust 為什麼不支援、卡在哪個資料結構或不變量、要改多大。
  - d. 加速候選：預期加速、會不會影響逐位元相同、風險。
- **最後要使用者決定的事**列在 §9。第 2 段（慢層上限的 Rust 版）的做法在 §10，**等使用者同意才做**。

## 1. 前提、版本、這台的狀態

**前提（第一份 prompt 的環境與對拍）：全部 PASS，我重新抽查過**
- STATUS_RUX 00:15 節的對拍，我逐一看了比對輸出：
  - G0 Python／Rust：`cells compared=128064 differing=0`（runs `20261106-235845-icpe-rux-compare-g0-py`、`20261107-000309-icpe-rux-compare-g0-rust`）。
  - xc_sa Python／Rust：`cells compared=1280 differing=0`（runs `20261107-000429-…-compare-xc_sa-py`、`20261107-000452-…-compare-xc_sa-rust`）。
  - npz 指紋：`DIGEST_RESULT IDENTICAL n=13 bad=0`（run `20261106-233920-icpe-rux-compare-npz-digest`）。
  - 雜訊 B1／B2：上一段的比對結果**沒有存檔**，我重跑同一支 `cmp_noise.py`：B1 40,426 格、B2 39,440 格，`differing=0`（run `20261107-001138-icpe-rux-recheck-noise`）。
- **發現的落差（不影響結論）**：上一段有 7 個 run 目錄沒有 `exit_code` 檔（`…-dl`、`…-convert-sa`、`…-g0-py`、`…-g0-rust`、`…-xcheck-xc_sa-py`、`…-xcheck-xc_sa-rust`、`…-p4-noise-subset`）。它們的比對 run 都有 exit 0。之後的 run 改用 `/root/icpe-next/tools/rux_run.sh`（複本在 `code/icpe/rux_x1fast/stage1/rux_run.sh`），一定寫 `exit_code`，並且每 5 s 記 MemAvailable 與 load（< 3 GiB 就砍掉整組）。

**pull**
- `paper-hierarchical-kv-state`：`git fetch` 後各分支的遠端都沒有新 commit（icpe-next-data 仍是 `8d70cbc`、icpe-next-sim `90dae78`、icpe-next-a `6b84a5c`、icpe-next-ana `43e7ffc`）。本機 `icpe-next-rux` 比 `origin/icpe-next-data` 多 2 個 commit，**還沒 push**。
- `/root/Proj/kvsweep`：fast-forward 到 `origin/main` `6d3c5f9`（第 1 段分析＋原型原始碼）。本報告引用它的 `docs/stage1/A_shapes.md`（以下 KA）、`B_profile.md`（KB），項目 1 的事實我全部回原始檔核對過，項目 2–5、7 核對了主要的出處與行號；數字以這台實測為準。

**依據的版本**

| 代號 | 檔案 | 版本 |
|:--|:--|:--|
| PR | `origin/icpe-next-a:docs/icpe_brief_20261030/next/PREREG.md` | `6b84a5c` |
| NP | `docs/icpe_brief_20261030/NEXT_PLAN_20261104.md` | HEAD（icpe-next-rux） |
| SS／SC | `origin/icpe-next-sim:…/next/STATUS_SIM.md`／`SIM_CHANGES.md` | `90dae78` |
| py | `code/p3x/x1_sim.py` | blob `8d0563ed`（icpe-next-sim 與本分支相同） |
| rs | `code/p3x/x1fast/src/main.rs`（v5） | tree `cef76d3f`（同上） |
| XR | `code/p3x/x1_rust.py` | 同上 |
| AN | `code/icpe/next_sim_analyze.py` | 同上 |

**這台**：RUX3DPRO，WSL2，Ryzen 7 7800X3D（8 核 16 執行緒），22 GiB，`perf` 沒裝（`perf_event_paranoid=2`）→ 硬體計數器一律 NOT_MEASURED，Rust 只用 x1fast 自己的引擎計時。

## 2. 總表

| # | 項目 | 登記 | Python 參考實作 | Rust 缺口（c） | 改動規模〔推算〕 | 這台量到的時間（b） |
|:--|:--|:--|:--|:--|:--|:--|
| 1 | 慢層容量上限 | 已登記（PR:32）；×7 配對與「先丟乾淨副本」是使用者 11/4 決定、**PREREG 還沒寫**；格子未定（PR:126） | **有**（py:521–695、947–1007） | 沒有慢層順序結構與用量計數；`parse_spec` 遇到多的欄位就停（rs:1476）；快速路徑會漏掉有副作用的 `_fix` | 約 350–450 行 | 見 §3b：Python 有上限 ÷ 沒上限＝1.3–2.5×（LRU、ARC、S3FIFO、GDSF；SemiAnalysis 全部規則）到 **7.8–125×**（LRU2、PENS、BEL，Mooncake／Bailian） |
| 2 | block 粒度 APC | 只承諾要做（PR:95）；語意未登記 | **沒有** | 「session 整個在同一層」：`sb[s]`＝請求序號、`st[s]` 單值、GPU LRU 以 session 為單位 | 約 400–600 行 | 對應 job＝GPU-LRU（§3b 的 nocap6） |
| 3 | 排隊 | 未登記（PR:57、126） | **沒有** | 「一次一個請求、不排隊」：照到達順序的 `step` 迴圈；所有時間設定共用一個狀態機 | 約 300–500 行 | 必須一個 job 一個時間設定：同樣工作量 Python 多 3.4–3.7×、Rust 多 3.2–4.3×（§5b） |
| 4 | W3 線索 (a) | 未登記（PR:57；NP:352 沒答案） | **沒有** | `Trace` 沒有工具／使用者旗標；GPU 規則只有 4 種（rs:1466） | 約 80–150 行 | 對應 job＝TTL（§6b） |
| 5 | W5 一次拆完 | 未登記（PR:57） | **沒有** | CPU 層常駐 BF16（rs 用到 `BF` 36 處）；還原逐塊付 d | 約 100–200 行 | 對應 job＝CPU 128／256 GiB（§7b） |
| 6 | W7 合成地圖 | 未登記；使用者定 ICPE 之後 | 不需要（缺的是產生器） | 引擎 0 行 | 產生器 NOT_MEASURED | — |
| 7 | STAIR | 不在範圍（舊 T3 敏感度） | 有，但不能開 GPU 層（py:360–362） | `View::restore` 只有 Cake／只載入（rs:2、129） | 約 100–150 行 | job 數 0 |

規模的〔推算〕：項目 1 是照 py 行數 × 約 1.3（Rust／Python 比，KA 用的經驗值）＋掛鉤與輸入輸出；項目 2–5、7 照 KA §10.2 的算法，我核對了它引用的 rs 行號。

## 3. 項目 1：慢層容量上限（主要項目）

### 3a. 形態

**一個 job**
- 輸入：一份紀錄變體（`nb:` 或 Mooncake）；CPU 規則、CPU 容量；GPU 層（111.0625 GiB、wb；規則 LRU／BEL／TTLA／TTLB）；**加上** `slow_cap_frac`（PR:32 定 0.5）與 1 個慢層規則（`SLOW_POLICIES`＝LRU、LRU2、ARC、S3FIFO、GDSF、PENS、BEL；py:517）。
- 上限＝`int(floor(frac × meta.n_blocks_full))` 個 INT4 塊（py:954）。
- 輸出：`x1_sweep.csv` 每個時間設定一列，多 5 欄 `slow_cap_frac`、`slow_policy`、`slow_cap_blocks`、`n_slow_clean_evict`、`n_slow_sess_drop`（py:2207–2211）；逐請求 npz 檔名多 `_sc0.5<規則>`（py:2220–2221）。
- 對應的 v5 job：同一組參數、沒有上限的 job。1 個 v5 job → 7 個新 job（每個慢層規則一個）。
- **job 怎麼拆**（py:2263–2276）：慢層 GDSF 和 CPU GDSF 一樣按 (慢層, GPU 閒／忙) 拆成 1 個時間設定一個 job；其他慢層規則跟原本的拆法（不依時間的規則一個 job 跑 6 個時間設定）。

**要跑多少**（`x1_sim.jobs_for` 直接數；run `20261107-001901-icpe-rux-count-jobs-sc`；CPU 9 個容量、慢層 ssd／b32／nfs × GPU 閒／忙、Cake）

| 規則集合 | 每個變體的 job | 每個變體的列 | 沒有上限的 job（v5） | 22 個有授權變體〔推算〕 | 40 個變體〔推算〕 |
|:--|--:|--:|--:|--:|--:|
| 完整：CPU 23 × GPU 16 × 慢層 7 | 129,924 | 139,104 | 18,342 | 2,858,328 | 5,196,960 |
| CPU 17 × GPU 16 × 慢層 7 | 95,796 | 102,816 | 13,518 | 2,107,512 | 3,831,840 |
| 只算 O-SLOW：NON_ORACLE 10 × GPU 不預測 15 × 慢層 7 | 54,540 | 56,700 | 7,740 | 1,199,880 | 2,181,600 |

- 每個慢層規則的列數都＝沒有上限的列數（完整集合：19,872）；job 數只有 GDSF 不同（19,872 對 18,342），因為它一個 job 一個時間設定。
- 只算 O-SLOW 那列用的是 AN:10–12、91–95 的定義：④＝NON_ORACLE × GPU 不預測 × 慢層 BEL；②＝NON_ORACLE × GPU 不預測 × 慢層 6 個不預測規則。
- 上面 3 列和 KA §3 的數字完全相同（獨立重算）。
- 加雜訊（`uses_oracle` 把慢層 BEL 算進去，py:2120–2124）或 P2／P3 的時間設定時，數字還要再乘；**範圍未定**。

**哪些已登記、哪些要先決定**
- 已登記（PR:32）：上限＝⌊0.5 × n_blocks_full⌋ 個 INT4 塊；② 用 LRU、LRU2、ARC、S3FIFO、GDSF、PENS 逐格取最快；④ 最晚回來的先丟；被丟的對話回來時全部重算；**跑在全部正式格子上（第二批）**；主結果仍是沒有上限。
- 使用者 11/4 決定、**PREREG 沒寫**（SS:151–158）：① 慢層上所有 INT4 塊都算進上限，超過時先丟「乾淨的副本」；② 慢層規則和 GPU、CPU 規則全部配對（×7）。SS:158 寫「是否要 h200 記成偏離由使用者決定」。
- **要使用者決定**：跑哪些格子（PR:126：「格子另外定、跑之前登記」），見 §9 D1-4。

### 3b. 時間花在哪（這台實測）

**怎麼量**
- 驅動程式：`code/icpe/rux_x1fast/stage1/prof_sc.py`（只 import x1_sim／x1_rust，不改它們）。
- 同一個 job：CPU LRU × GPU-LRU wb 111.0625 GiB、CPU 111.0625 GiB（代表格容量）、Cake。
  - `nocap6`＝沒有上限、6 個時間設定（v5 支援的對應 job）；`nocap1`＝沒有上限、只有 ssd idle（慢層 GDSF 的對應 job）。
  - `sc:<規則>`＝frac 0.5 加慢層規則；GDSF 只有 1 個時間設定，其他 6 個。
- 單執行緒、`nice -n 19`、交錯重複（Mooncake 5 次、Bailian 3 次、SemiAnalysis 1 次），報中位數。Python＝`time.process_time()`；Rust＝x1fast 批次模式自己量的引擎時間（不含讀紀錄）。
- 三份紀錄**同時**各一個行程跑（3 個單執行緒行程，另有 job 形狀的計時同時在跑）；1 分鐘 load 1.1–3.5／16 執行緒，每筆都記在 CSV。
- runs：Mooncake `20261107-001735-icpe-rux-prof-sc-mc`、Bailian `20261107-001737-icpe-rux-prof-sc-bl`、SemiAnalysis `20261107-001739-icpe-rux-prof-sc-sa`。
- 紀錄：Mooncake conversation R1（12,031 請求、8,520 session）、Bailian coder（43,011、26,406；這台新轉的 npz，指紋 `192949471c…` 和 acvlab 的 `npz_digest.csv` 相同，runs `20261107-001533-…-convert-bailian-coder`、`20261107-001551-…-npz-digest-bailian-coder`）、SemiAnalysis N16 s1（98,827、2,090）。
- **TraceLab：NOT_MEASURED**。這台沒有 TraceLab 的 npz；acvlab 的準備階段單一行程最大 RSS 14,890,176 KiB＝14.2 GiB（SS:394；單位更正在 SS:493），這台只有 22 GiB。要不要在這台試，見 §9 D1-6。

**結果 1：每個 job 的時間（秒，中位數）**

括號裡是「有上限 ÷ 沒上限」（GDSF 對 nocap1，其他對 nocap6）〔算術〕。Mooncake 5 次、Bailian 3 次、SemiAnalysis 1 次；每個設定各次的逐請求指紋都相同。

| 設定（時間設定數） | Mooncake conversation R1 | Bailian coder | SemiAnalysis N16 s1 |
|:--|--:|--:|--:|
| Python nocap6（6） | 0.268 | 0.608 | 13.19 |
| Python nocap1（1） | 0.165 | 0.361 | 7.51 |
| Python sc:LRU（6） | 0.351（1.31×） | 0.842（1.39×） | 24.74（1.88×） |
| Python sc:ARC（6） | 0.375（1.40×） | 0.851（1.40×） | 24.63（1.87×） |
| Python sc:S3FIFO（6） | 0.374（1.40×） | 0.859（1.41×） | 24.50（1.86×） |
| Python sc:GDSF（1） | 0.302（1.83×） | 0.628（1.74×） | 18.92（2.52×） |
| Python sc:LRU2（6） | 2.085（**7.8×**） | 15.41（**25.4×**） | 24.77（1.88×） |
| Python sc:PENS（6） | 9.307（**34.8×**） | 61.16（**100.6×**） | 25.87（1.96×） |
| Python sc:BEL（6） | 11.03（**41.2×**） | 76.17（**125.4×**） | 24.96（1.89×） |
| Rust v5 nocap6（6） | 0.0134 | 0.0298 | 0.298 |
| Rust v5 nocap1（1） | 0.0096 | 0.0167 | 0.159 |

上限真的有作用（引擎計數，Python；「清」＝清掉乾淨副本、「丟」＝丟掉慢層 session）：

| 慢層規則 | Mooncake（上限 85,449 塊） | Bailian（78,541） | SemiAnalysis（355,025） |
|:--|:--|:--|:--|
| LRU | 清 34,158／丟 3,896 | 清 38,735／丟 11,508 | 清 157,921／丟 193 |
| LRU2 | 34,038／4,249 | 38,420／14,498 | 157,773／189 |
| ARC | 33,991／4,271 | 38,692／13,834 | 168,739／361 |
| S3FIFO | 33,991／4,271 | 38,692／13,834 | 168,739／361 |
| GDSF | 34,263／4,966 | 38,001／13,765 | 175,223／819 |
| PENS | 40,265／7,456 | 41,349／18,587 | 175,899／938 |
| BEL | 34,941／3,872 | 40,089／11,361 | 157,921／193 |

④ 那一邊的組合（CPU LRU × **GPU BEL** × 慢層 BEL；Mooncake 3 次，run `…-prof-sc-mc` 的 `out_gbel`）：Python 沒上限 4.12 s → 有上限 12.06 s（2.93×）；Rust v5 沒上限 0.192 s（Python ÷ Rust 21.5×，指紋相同）。慢層 BEL 丟 3,238 次 × 平均 3,570 個慢層 session＝1,156 萬次 `next_req`。Bailian（3 次，`…-prof-sc-bl` 的 `out_gbel`）：Python 沒上限 7.67 s → 有上限 74.87 s（9.77×）；Rust v5 沒上限 0.160 s（Python ÷ Rust 48.0×，指紋相同）；慢層 BEL 丟 11,115 次 × 平均 10,973 個＝1.22 億次 `next_req`。

**結果 2：Python 和 Rust v5 的輸出相同、慢多少**
- 沒有上限的設定，Python 與 Rust v5 的逐請求陣列（ttft、rest、nC、nS、nR）sha256 **全部相同**（3 份紀錄 × nocap6／nocap1）。
- Python ÷ Rust v5（沒有上限）：Mooncake 20.0×（nocap6）／17.2×（nocap1）、Bailian 20.4×／21.6×、SemiAnalysis 44.3×／47.3×；加上 §6b、§7b 的 job 形狀，這台的範圍是 **17–50 倍**。prompt 寫的「30–90 倍」在這台偏低：小紀錄（Mooncake、Bailian）倍數最低。

**結果 3：時間花在哪（cProfile；開 profiler 的那一次另外跑、不計入上表）**
- **SemiAnalysis：成本來自「關掉快速路徑」，不是慢層規則**
  - 7 種慢層規則都慢 1.86–1.96×（GDSF 對 nocap1 2.52×）。
  - `_fix` 的呼叫數從 13,250,323（沒有上限，`_fix_g`）變成 46,620,252（`_fix_sc`），×3.52。原因：開上限時 `_fix_fast=False`（py:961），回合結束不再跳過「互相抵銷的前綴」（py:1675、1795、1891），每塊都呼叫 `_fix`（py:1710、1852、1926）。
  - `_slow_holders` 36,340,848 次、`_slow_enforce` 98,827 次（每個請求一次）。
  - 慢層真的丟 session 只有 193 次（LRU、BEL），丟的時候慢層平均 1,896 個 session → 掃描成本可以忽略。
  - cProfile 自身時間占比（sc:LRU）：`_fix_sc` 24.4%、`_turn_end_wb` 17.0%、`restore` 15.1%、`_fix_g` 10.2%、`_slow_holders` 7.3%、`_slow_enforce` 5.8%。
- **Mooncake、Bailian：成本來自 LRU2、PENS、BEL 每次丟 session 都掃過全部慢層 session**
  - `SlowOrder.select` 對這三個規則是 `min(self.mem, key=…)`／`max(self.mem, key=…)`（py:633–644），每次 O(慢層 session 數)。
  - Mooncake BEL：丟 3,872 次 × 平均 4,585 個慢層 session＝**17,751,636 次 `next_req`**；cProfile 裡 `next_req`、`excl_start`、`next_access` 三個合計占 58%。
  - Bailian BEL：丟 11,361 次 × 平均 11,381 個慢層 session＝**129,302,135 次 `next_req`**；開 cProfile 時這一個 job 要 204 s，`next_req`、`excl_start`、`next_access` 三個自身時間合計占 60%。
  - PENS、LRU2 沒有開 cProfile 計次；它們在 Bailian 丟 18,587、14,498 次，丟的時候的慢層 session 數應和 LRU（平均 11,216）同級〔推論〕。
  - **Rust 照抄後掃描還剩多貴**〔推算〕：用 v5 自己的 GPU-BEL 反推每個成員的成本——Mooncake GPU-BEL 的 Python 呼叫 7,390,848 次 `_next_req_bl`；Rust v5 GPU-BEL job 0.192 s − GPU-LRU job 0.013 s ≈ 0.18 s → **每次約 24 ns**（含 v5 每次複製成員清單的成本，是上限）。
    - Mooncake 慢層 BEL：1,775 萬 × 24 ns ≈ 0.43 s（Python 11.0 s 的 1／26；v5 沒上限 job 0.013 s 的約 32 倍）。
    - Bailian 慢層 BEL：1.29 億 × 24 ns ≈ 3.1 s（Python 76.2 s 的 1／25；v5 沒上限 job 0.030 s 的約 104 倍）。
    - 所以照抄版已經比 Python 快 20 倍以上，但 Mooncake／Bailian 的 LRU2、PENS、BEL 會遠慢於 v5 的一般 job → §3d 候選 1–3。
  - LRU、ARC、S3FIFO、GDSF 是 O(1) 或 O(log n)，只多 `_fix_sc` 這層：Mooncake sc:LRU 的 `_fix_sc` 442,065 次（沒有上限的 `_fix_g` 270,716 次，×1.63）。
- **第一次超過上限的位置 r\***（run `20261107-002132-icpe-rux-first-overflow`；慢層 LRU、1 個時間設定）

  | 紀錄 | 請求數 | 上限（塊） | r\* ÷ 請求數（CPU 8／32／111／256／1,024 GiB） |
  |:--|--:|--:|:--|
  | Mooncake conversation R1 | 12,031 | 85,449 | 0.449／0.451／0.456／0.466／0.537 |
  | Bailian coder | 43,011 | 78,541 | 0.481／0.482／0.488／0.499／0.571 |
  | SemiAnalysis N16 s1 | 98,827 | 355,025 | 0.901／0.901／0.904／0.906／0.940 |

  - r\* 之前，7 種慢層規則與沒有上限的 job 狀態完全相同（上限沒動作時，`SlowOrder` 和 `clean_q` 只是旁邊的記錄）〔推論，照 py:988–1007〕。這是 §3d 候選 4 的依據。

**結果 4：一些觀察（不是錯誤，但要讓使用者知道）**
- ARC 與 S3FIFO 的逐請求輸出 sha256 在 3 份紀錄都相同（Mooncake `51bd1921…`、Bailian `260d04fe…`、SemiAnalysis `2c14b698…`）。SemiAnalysis 上 BEL 和 LRU 也相同（`08106388…`）。
  - 〔推論〕兩種規則在這些 job 裡都退化成「丟最早進慢層、而且沒回來過的 session」：ARC 的 T1 與 S3FIFO 的 S 都是 FIFO，回來過的 session 都被保護（ARC 進 T2、S3FIFO 進 M）。SemiAnalysis 被丟的 session 大概都不會再回來，所以 BEL 和 LRU 選到同一批。只看了 1 個 job，**不能推到全部格子**。
  - 對 Rust 版沒有影響：照抄就會得到同樣的結果。
- 開上限後，Mooncake／Bailian／SemiAnalysis 每個 job 的上限都真的有作用（丟 session 193–18,587 次、清乾淨副本 33,991–175,899 次），所以這些子集能測到上限的程式路徑。

**換算到正式量〔推算〕**
- 使用者寫的「Python 在這台跑 TraceLab 一個 block × 7 種慢層規則約 4 天」：
  - 算式＝TraceLab c1 一個 P0 block 的 Python worker 秒 × 7 種規則的平均倍數 ÷ worker 數。
  - Python TraceLab c1 block（6,174 個 job）在 acvlab 是 sweep 19,670 s × 40 worker（SS:346）；這台每個 worker 約快 2.3×（G1 在 acvlab 6,453 s × 4 worker＝25,812 worker 秒，SS:342；這台 G0 是 922.9 s × 12＝11,075 worker 秒，STATUS_RUX）〔推算，負載不同〕→ 這台約 34 萬 worker 秒。
  - 如果 TraceLab 像 SemiAnalysis（每種規則約 2×）：34 萬 × 7 × 2 ÷ 12 worker ≈ 40 萬 s ≈ **4.6 天**，和使用者的估計一致。
  - 如果像 Bailian（LRU2、PENS、BEL 掃描很重）：會長很多。TraceLab 是 8,058 個 session 的長對話，比較像 SemiAnalysis〔推論〕，但 NOT_MEASURED。

### 3c. 現有 Rust 為什麼不支援、要改多大

**卡在哪**
- 入口：`x1_rust.unsupported()` 對有上限的 job 回傳 `"slow_cap"`（XR:101–104）→ 該 job 退回 Python；`parse_spec` 遇到第 14 欄以後不是 `nz=` 就 panic（rs:1476）。
- 資料結構：`Eng` 只有 `on_slow: Vec<bool>`（rs:426），沒有慢層用量計數、沒有慢層順序結構、沒有 `clean_q`。
- **不需要打破任何不變量**：session 粒度、`sb[s]`＝請求序號（rs:427、455）都照舊。

**要改的地方（照 Python 的呼叫點逐一對應）**

| Python | 做什麼 | Rust 對應 |
|:--|:--|:--|
| `SlowOrder`（py:521–695，175 行） | 7 種規則的 add／remove／select | 新 struct。`mem`＝`OList`（插入順序＋`move_to_end`，rs:272）；ARC／S3FIFO 用 `OList`＋位元組計數（照 CPU 版 rs:675–722 的寫法）；GDSF 用 `BinaryHeap<Reverse<(F, last, ver, s)>>`（同 rs:436 的 gheap）；LRU2／PENS／BEL 照 Python 的走訪順序掃 |
| `_sc_init`（py:947–965） | 上限、計數、掛鉤 | `new_eng` 多一個 `sc: Option<SlowCap>`；上限由 Python 算好傳進來（避免兩邊各算一次 floor） |
| `_fix_sc`（py:973–981） | n_slow 增減；變乾淨就進 `clean_q` | `fix()`（rs:461）尾端加一段；`_slow_holders`＝ref_live − ref_cpu −（wb 時）ref_gpu，條件和 rs 的 `fix_g` 相同 |
| `_demote_sc`（py:983–986） | 降級後 `sorder.add` | `demote()`（rs:738）尾端 |
| `_enforce_sc`／`_slow_enforce`（py:988–1007） | CPU enforce 之後檢查上限：先清 `clean_q`，再照規則丟 session | `enforce()`（rs:815）尾端；丟 session 用 `drop_()`，**它的回傳值丟掉不算寫入**（Python 也丟掉，py:1006） |
| `drop`（py:1471–1472） | 慢層狀態的 session 被丟時離開結構 | `drop_()`（rs:762）開頭 |
| `step`（py:1638–1639） | 慢層狀態的 session 回來＝命中 | `step()`（rs:1139）在 `cur` 設好之後、分 GPU 路徑之前 |
| `_fix_fast=False`（py:961） | 關掉三個回合結束的快速路徑 | **rs 的三處都要改**：`step` 的 k0（rs:1175–1177）與「只在必要時才 fix」（rs:1198）、`turn_end_wb`（rs:1262–1264、1301–1302）、`turn_end_wt`（rs:1343–1345、1366）。開上限時 k0＝0、每塊都 `fix` |
| `audit`（py:1997–2000） | n_slow＝sum(on_slow)、n_slow ≤ 上限或結構已空、結構成員＝S_SLOW 的 session | 驗證模式的斷言 |
| `_job_rows`（py:2207–2211） | 輸出 3 個計數 | `write_out`（rs:1543）另寫一個檔（例如 `eng_sc.i64`），**不動** `eng.i64` 的 7 個值，讓 `read_outputs` 的 `assert len(ei) == 7`（XR:393）在沒有上限時照舊 |

**最容易出錯的地方（第 2 段要專門驗）**
1. **`clean_q` 的順序取決於每一次 `_fix` 呼叫**。Rust v5 的快速路徑會少叫 `_fix`（rs:1198、1302、1366 只在「沒人持有」或「CPU 沒有持有者卻有 BF16」時才叫）。沒有上限時那些呼叫沒有副作用，但開上限後每次呼叫都可能把塊放進 `clean_q`。少一次就可能改變之後哪個副本先被清 → 結果不同。所以照 Python 一樣全部呼叫。
2. **wb 回合結束的順序**：Python 慢路徑是「全部遞減 → `_g_release(old)` → 全部 `_fix`」（py:1846–1852），Rust 是逐塊合在一起（rs:1290–1305）。`_fix`／`_slow_holders` 只讀寫該塊自己的 refcount（SC:114 的理由），所以 `clean_q` 的內容和順序相同〔推論〕；第 2 段用逐請求比對證實。
3. **走訪順序與同分**：Python `min`／`max` 回傳**第一個**最小／最大；Rust 的 `max_by_key` 回傳**最後一個**。LRU2、PENS、BEL 的鍵都含 `last[x]`，而每個請求只屬於一個 session，所以慢層裡的 session 的 `last` 互不相同 → 不會同分〔推論〕。照樣在驗證模式裡斷言「沒有同分」。
4. **雜訊 B2 的亂數順序**：慢層 BEL 的 `max(self.mem, key=next_req…)` 每個成員都呼叫一次 `_nz`、每次都抽亂數（py:1034–1046）。Rust 必須照 `mem` 的插入順序走訪、每個成員呼叫一次，不能跳過或換順序。
5. **GDSF 的成本表**：慢層 GDSF 的 `g_c[i] = max(f_i − (i4 ÷ slow_r + d), 0)`（py:539–541），和 CPU GDSF 的 `max(min(f_i, l_sl) − l_cpu, 0)`（py:748–751）不同；要另外一份。`slow_r` 用 `views.txt` 傳進來的 `v.slow.r`＝`K.slow.r × GiB`（py:206、253），和 Python 是同一個 double。
6. **ARC 的浮點**：`p` 的更新有整數除法轉浮點（`b2b / max(b1b, 1)`，py:562、567），Rust 要用 `as f64` 再除；整數在 2^53 以下時兩邊是同一個 double（慢層位元組最多＝上限 × 每塊 INT4 位元組 ＜ 355,025 × 64 MiB ≈ 2.4×10^13，ARC 的鬼表最多再 2 倍〔推算〕，都遠小於 2^53≈9.0×10^15）。

**規模〔推算〕**：SlowOrder 約 230 行＋掛鉤與快速路徑的閘門約 60 行＋輸入輸出約 40 行＋驗證模式約 50 行＝**約 380 行**；`x1_rust.py` 的對應改動（`unsupported` 放行、`write_job` 加 `sc=` 欄、`read_outputs` 讀 `eng_sc.i64`）約 30 行，放在我自己的目錄的複本裡，正式整合由 icpe-sim 做。

### 3d. 加速候選（第 2 段先做逐位元相同的照抄版；下面是之後可以加的）

| # | 候選 | 預期效果〔推算〕 | 逐位元相同？ | 風險 |
|:--|:--|:--|:--|:--|
| 0 | **照抄的 Rust 版**（第 2 段要做的） | 取代 Python 退回。沒有上限時 Python 是 Rust 的 17–50 倍；LRU、ARC、S3FIFO、GDSF 若保持同樣比例，Mooncake sc:LRU 約 0.35 s → 約 0.02 s。LRU2、PENS、BEL 在 Mooncake／Bailian 受掃描限制，約比 Python 快 25 倍（§3b 結果 3） | 是（目標） | 中：見 3c 的 6 個易錯點 |
| 1 | **BEL 用 lazy heap**（照 kvsweep C 的 P5c：按共用組的 epoch 失效＋「下次回來」到期的醒來 heap） | 照抄版 Bailian 慢層 BEL 約 3.1 s／job、Mooncake 約 0.43 s〔推算，每次 24 ns〕，是 v5 沒上限 job 的 30–100 倍；heap 版把掃描降到每次丟 O(log n) | 是（鍵含 −last 不會同分；但 **只限沒有雜訊**：B2 每次問都抽亂數；B1 的有雜訊的值會被夾到至少 `cur+1`，模型 M 還用到 `t[cur]`，鍵都隨 `cur` 變（py:1038–1066）→ 有雜訊時只能全掃） | 中高：要證明鍵變大的時機全部被捕捉；用驗證模式（每次 select 都和全掃比） |
| 2 | **LRU2 用有序集合** | 鍵（prev 或 −1, last）在 session 待在慢層期間不變（prev／last／nreq 只在回合結束改，那時 session 已離開慢層）→ 每次丟 O(log n) | 是 | 低 |
| 3 | **PENS 的成本快取** | 鍵＝成本 ÷ (cur − last + 1)，隨 `cur` 變，不能用 heap；但「成本」只在共用組的 refcount 變時才變 → 用 rs 現有的 group epoch（rs:446–449）快取，掃描時每個成員只剩一次除法和比較 | 是 | 低中 |
| 4 | **r\* 前共用**：同一組 (CPU 規則, GPU 規則, CPU 容量) 的 7 個慢層規則（加上沒有上限的那個 job），在第一次超過上限之前狀態完全相同 | 7 個 job 的成本 ≈ r\*／n ＋ 7 ×（1 − r\*／n）個 job（共用段只跑一次）：SemiAnalysis r\*／n≈0.90 → 約 1.6 個 job 換 7 個（≈4.4×）；Mooncake／Bailian r\*／n≈0.45–0.57 → 約 3.6–4.3 個換 7 個（≈1.6–2.0×）。r\* 只在 CPU LRU × GPU-LRU 量過，其他規則組合 NOT_MEASURED | 是（共用段同時維護 7 個 SlowOrder 的 add／remove，r\* 時複製） | 中：要改 job 的打包（x1_sim／x1_rust 是 icpe-sim 的）；GDSF 是 1 個時間設定，要從 6 個時間設定的共用段裡只取一個 |
| 5 | **快速路徑＋補記 `clean_q`**：開上限時仍跳過互相抵銷的前綴，但對前綴每塊補做「在慢層、沒有慢層持有者就放進 `clean_q`」 | SemiAnalysis 的 `_fix` 呼叫 ×3.52 是它變慢的主因；補記只讀不寫 | 是〔推論：前綴塊的 +1／−1 抵銷後狀態等於回合前，`_fix` 除了放進 `clean_q` 不做別的，SC:124 的理由〕 | 中：要逐請求證明 `clean_q` 的內容與順序相同 |

- 候選 1–5 都**不改語意**，但每一個都要再做一次逐位元對拍。建議第 2 段只做 0；1–5 要不要做、做哪幾個，見 §9 D1-3。
- 不建議：抽樣或近似（SHARDS 類）——不是逐位元相同。

## 4. 項目 2：block 粒度 APC

**a. 形態**
- 一個 job＝主結果的 job，GPU 層規則多一個 APC。對應的 v5 job＝GPU-LRU：SS:68「session 粒度下 APC 和 GPU-LRU 完全相同」。
- APC 不依時間 → 一個 job 跑 6 個時間設定。job 數（KA §4，用 GPU-LRU 代替 APC 數的）：只放進 ②（NON_ORACLE × APC）每個變體 180；② 與 ④ 都放（CPU 17）333；CPU 23 477。
- 登記：PR:95 只登記「做好後加進 ②、受影響的格子全部重算」；**確切語意、範圍都沒登記**。描述只有 SS:32「從最久沒用的 session 尾端一塊一塊逐出，可以留半個 session」。

**b. 時間（這台）**：對應 job 就是 §3b 的 `nocap6`：Rust v5 Mooncake 0.0134 s、Bailian 0.0298 s、SemiAnalysis 0.298 s。APC 本身的成本 NOT_MEASURED（沒有程式）。

**c. Rust 缺口**
- 卡住的不變量是「session 整個在同一層」：`sb[s]`／`gsb[s]` 是請求序號（rs:427、441），session 持有的塊＝整個請求；`st[s]` 只有一個值；GPU LRU 以 session 為單位（`glru: OList`）；`g_evict_wb`（rs:960）整個 session 寫進 CPU。
- APC 要：每個 session 在 GPU 上的前綴長度 `gn[s]`；wb 下前段在 GPU、尾段在 CPU；CPU 那邊的大小與 ARC／S3FIFO／GDSF 的位元組要改成部分 session。
- 還原不用動：`View::restore` 已經吃「前 g 塊在 GPU」（rs:131；`step_gpu` 數 g，rs:1229–1230）。
- 規模〔推算〕約 400–600 行（`g_*` 約 100 行、`turn_end_wb`／`turn_end_wt` 約 170 行大致重寫，加 CPU 那邊）。**Python 參考實作也沒有**（SS:32 估 Python 版「多約 1 天」）。

**d. 加速**：沒有可比的基準（新功能）。成本預期和 GPU-LRU 同級，再加逐塊逐出的迴圈〔推論〕。

## 5. 項目 3：排隊

**a. 形態**
- NP:266：「W2 若用固定並行數，模擬器要加排隊；做不到就寫成『上限』」；Python 現在是「一次一個請求、GPU 不排隊」。
- 加排隊後，回合結束的時刻取決於該時間設定的 TTFT → 不同時間設定的事件順序不同，**一個 job 只能跑 1 個時間設定**〔推論〕。job 數＝v5 的列數 × q（q＝排隊設定數，未定）：每個變體 19,872 q（KA §5）。
- 登記：**未登記**（PR:57、126）。

**b. 時間（這台）**：「一個 job 一個時間設定」本身的代價可以量：同一個 job 拆成 6 個 1 個時間設定的 job，總時間是原本的 6 × nocap1 ÷ nocap6 倍。
- Python：Mooncake 3.70×、Bailian 3.56×、SemiAnalysis 3.42×；Rust v5：Mooncake 4.28×、Bailian 3.36×、SemiAnalysis 3.20×〔算術，§3b 的 nocap1／nocap6 中位數〕。
- 排隊本身的成本 NOT_MEASURED（沒有程式）。

**c. Rust 缺口**：`run_job_dir` 照到達順序 `for r in 0..n { step(r) }`（rs:1571）；`step` 同一步做完還原與回合結束（rs:1139–1224）；所有時間設定共用一個狀態機、只有 I/O 線分開。要改成事件驅動（到達與完成同一個 heap、完成時才改狀態），`step`／`step_gpu` 拆兩半，`turn_end_*`、預取推進、`g_expire` 都要動。規模〔推算〕約 300–500 行。**Python 參考實作也沒有**，模型也沒定。

**d. 加速**：一個 job 一個時間設定 → 沒有共用狀態機的好處；可以共用的只有讀紀錄。

## 6. 項目 4：W3 線索 (a)

**a. 形態**
- NP:42 與 NP:320 的寫法不同：「留在／搬回 CPU」對「留在 GPU／搬回 CPU」。產出＝「線索拿走完美預取改善量的比例」（NP:43）。
- 對應的 v5 job＝同一 job 的 GPU 規則換成 LRU 或 TTL；線索規則不依時間〔推論〕。
- job 數（KA §6）：③ 只配 CPU LRU 每個變體 9 個 job；配 NON_ORACLE 10 個每個變體 180 個；× 線索語意數 c（1 或 2）。
- 訊號：nb 檔有 `user_trigger`（py 用的 `x1_nb.py`）；Mooncake 的 `x1_trace.py` 沒有這個欄位（KA §6，我沒有重查 Mooncake 原始檔）。
- 登記：**未登記**（PR:57；NP:352「要不要列進 ICPE」沒有答案）。

**b. 時間（這台）**：最接近的 v5 job＝CPU LRU × GPU-TTLA:10（1 個時間設定，CPU 111.0625 GiB；3 次中位數；run `20261107-002255-icpe-rux-prof-shapes`，load 2.7–3.4）：

| 紀錄 | Rust v5（s） | Python（s） | Python ÷ Rust | 逐請求指紋 |
|:--|--:|--:|--:|:--|
| Mooncake conversation R1 | 0.00964 | 0.1795 | 18.6× | 相同 |
| Bailian coder | 0.01804 | 0.4192 | 23.2× | 相同 |
| SemiAnalysis N16 s1 | 0.30118 | 15.1985 | 50.5× | 相同 |

- 線索規則本身的成本 NOT_MEASURED（沒有程式）。

**c. Rust 缺口**：`Trace`（rs:394–397）沒有每個請求的工具／使用者旗標，`prepare_trace` 的 CSR 也沒寫；GPU 規則只有 LRU、BEL、TTLA、TTLB（rs:1466）。不用打破不變量。要改 `g_add`、`g_select`、`g_expire`、`turn_end_*` 的釘住判斷。規模〔推算〕約 80–150 行。**Python 參考實作也沒有**。

**d. 加速**：預期和 TTL job 同級〔推論〕。

## 7. 項目 5：W5（CPU 層一次拆完）

**a. 形態**：NP:50–53「CPU 層允許 INT4、用 M3-g 的一次拆完常數；只重跑 128／256 GiB 的代表格」；事先登記的預期「上限下降或不變」（NP:53）。job 數（KA §7）：CPU 23 × GPU 16 × CPU 128／256 每個變體 4,076 個；代表格 7 個變體（有授權 5 個）。**未登記**（PR:57）。

**b. 時間（這台）**：對應 v5 job＝CPU LRU × GPU-LRU、CPU 128／256 GiB、6 個時間設定（3 次中位數；run `20261107-002255-icpe-rux-prof-shapes`，load 2.1–3.3）：

| 紀錄 | CPU | Rust v5（s） | Python（s） | Python ÷ Rust | 逐請求指紋 |
|:--|--:|--:|--:|--:|:--|
| Mooncake conversation R1 | 128 | 0.01406 | 0.2640 | 18.8× | 相同 |
| Mooncake conversation R1 | 256 | 0.01131 | 0.2581 | 22.8× | 相同 |
| Bailian coder | 128 | 0.03049 | 0.6334 | 20.8× | 相同 |
| Bailian coder | 256 | 0.02387 | 0.6059 | 25.4× | 相同 |
| SemiAnalysis N16 s1 | 128 | 0.29552 | 12.9012 | 43.7× | 相同 |
| SemiAnalysis N16 s1 | 256 | 0.29677 | 13.0952 | 44.1× | 相同 |

- 一次拆完本身的成本 NOT_MEASURED（沒有程式）。

**c. Rust 缺口**：CPU 層常駐 BF16（`fmt=1`；回合結束、GPU 逐出時 `cpu_used += BF`），INT4（`fmt=2`）只出現在預取；還原時每個 INT4 塊付一次 d（rs:120、156）。要改 `View::restore`（整段一次付常數）、`m_cpu`、`fix`、`turn_end_*`、`g_evict_wb`、`demote`（已是 INT4 不必再量化）、GDSF／PENS 的成本表（rs:1500–1503 用 BF 算 l_cpu）。規模〔推算〕約 100–200 行。**Python 參考實作也沒有**。

**d. 加速**：沒有新的資料結構；成本和現在同級〔推論〕。

## 8. 項目 6、7

- **W7（合成參數地圖）**：使用者已定 ICPE 之後做（STATUS_DATA 09:05）。引擎不用改（Rust 吃任何紀錄的 CSR）；缺的是產生器（`code/p3_data/synth.py` 輸出的是 phase 3 的 parquet，沒有 512-token 塊的鍵鏈）。維度與抽樣設計的規模分析在 kvsweep `docs/stage1/A_shapes.md` §8（W7-1 到 W7-9 的待決事項），這段沒有重做。
- **STAIR**：只有舊 T3 敏感度用（`code/icpe/t3.py`）；Python 的 STAIR 不能開 GPU 層（py:360–362），這一輪的正式格子全部開 GPU 層 → **job 數 0**，不在範圍內。

## 9. 要使用者決定的事

**D0（跨項目 2–5）：Python 參考實作由誰寫？**
- 「逐位元相同」要先有參考實作。項目 2–5 的 Python 都沒有。
- 選項：(a) icpe-sim 寫 Python（它擁有 `code/p3x/*`），rux 寫 Rust 再對拍——和 v3–v5 同一套流程；(b) rux 在自己的目錄另寫 Python 參考版，再交 icpe-sim 審查併入。
- 建議 (a)：規則是「不改別人的檔」，而且參考實作和 Rust 由不同人寫，對拍才有意義。

**項目 1（慢層上限）**
- **D1-1 語意確認：「被丟的對話回來時全部重算」**。
  - PR:32 字面是「全部重算」。Python 的做法（py:1468–1501、1640–1656）：丟掉時把這個 session 的每塊 refcount −1，**別人還持有的共用前綴塊照常留著**；回來時照各塊的實際位置還原，遇到第一個沒存的塊才開始重算。所以「全部重算」只對沒有別人持有的塊成立〔推論，照程式〕。
  - 第 2 段會照 Python 一字不改。**要確認**：這是 PREREG 的意思嗎？（我認為是：共用前綴的資料還在，不該假裝不在。）
- **D1-2「先丟乾淨副本」「×7 配對」要不要請 h200 寫進 PREREG**（SS:158）。不影響第 2 段的程式（兩者 Python 都已實作）。
- **D1-3 第 2 段的範圍**：
  - (a) 只做照抄的 Rust 版（3d 的候選 0），逐位元相同就交給 icpe-sim；加速另開一段。**建議**。
  - (b) 照抄版過了之後，接著做 3d 的候選 2、3（低風險）；1、4、5 另外決定。
- **D1-4 要跑的格子**（PR:126：跑之前登記；第 2 段不需要先定）：
  - 規則集合：完整 129,924／CPU 17 版 95,796／只算 O-SLOW 54,540（每個變體 job 數，§3a）。
  - 變體：全部 40 個，或有授權的 22 個，或代表格（PR:118 的 7 個）。
  - 要不要加 P2（nvme807、只載入）、P3（wt、預取進 GPU）的時間設定？要不要和 W4 雜訊交叉？
- **D1-5 第 2 段驗證要不要也涵蓋 wt 與預取進 GPU**：Python 這兩條路徑都支援慢層上限。我的預設：**要**（Rust 照抄全部路徑，子集裡各放一格）。沒意見就照這樣做。
- **D1-6 TraceLab 要不要在這台量**：準備階段在 acvlab 單一行程最大 RSS 14.2 GiB（SS:394、493），這台 22 GiB、MemAvailable 約 20 GiB。可以開記憶體監看試一次（< 3 GiB 自動停），或不量、照 SemiAnalysis 推。prompt 的第 2 段驗證只要 Mooncake、SemiAnalysis、Bailian，所以這項不卡第 2 段。

**項目 2 APC**（選項照 KA §11，我核對過出處）
- 2-1 逐出的單位與順序：(a) 以 session 的 LRU 為序，從尾端一塊一塊逐出到夠為止（SS:32 字面）；(b) 全域 block LRU（每塊最後被讀的時間），同一請求裡尾端先出（像 vLLM）。
- 2-2 wb 下逐出的尾段：(a) 寫進 CPU（PR:54 的資料流）；(b) 直接丟。
- 2-3 wt（敏感度）下的定義。
- 2-4 塊大小：512（NP:269 保留 512）或 16（vLLM）。
- 2-5 APC 要不要也放進 ④ 的 GPU 不預測規則（偏離 6 的公平性理由，PR:176–183）。
- 2-6 「受影響的格子全部重算」（PR:95）指哪些：主結果；再加 P2／P3；再加 P4、慢層上限。
- 2-7 TTL 要不要也改成 block 粒度（PR:30 的 TTL 語意以「這個對話的 KV」為單位，是 session 粒度）。

**項目 3 排隊**
- 3-1 模型：(a) 單一 GPU、先到先服務、服務時間＝還原＋prefill；(b) 連續批次（要定批次規則）。
- 3-2 服務時間要不要算解碼（NP:276「完成時間要模擬解碼，不做」）。
- 3-3 狀態何時改：(a) 服務完成時；(b) 到達就改（等於沒有排隊對快取的影響）。
- 3-4 範圍：只跑固定並行的紀錄（SemiAnalysis；CacheWise 不在這台）或全部；主結果還是敏感度。
- 3-5 q：排隊設定的個數與數值。

**項目 4 W3 線索 (a)**
- 4-1 線索做什麼：(a) 留在 GPU、釘到回來；(b) 搬回 CPU；(c) 兩個都跑（NP:42 與 NP:320 寫法不同）。
- 4-2 「工具開始了」怎麼判斷：下一個請求的 `user_trigger=False`；Mooncake toolagent 沒有這個欄位，要排除還是另外定義。
- 4-3 指標：NP:43 的 (T① − T③) ÷ (T① − T④,O-PF)；「拿走大部分」的門檻沒定（PR:21）。
- 4-4 ③ 配哪些 CPU 規則：只配 LRU，或 NON_ORACLE 取最快。
- 4-5 範圍；4-6 要不要列進 ICPE（NP:352）。

**項目 5 W5**
- 5-1 一次拆完的常數怎麼放進模型：每次還原付固定值，或依長度換算（M3-g 量的長度待確認）。
- 5-2 CPU 層格式：一律 INT4，或照規則選；GPU 逐出時要不要付量化時間。
- 5-3 每層的還原方式（CPU 全部載入＋一次拆完、慢層 Cake？）。
- 5-4 CPU 容量：NP:52 的 128／256，或偏離 4 代表格的 111.0625／128（PR:118）；用哪幾個變體。
- 5-5 當主結果的替代，還是敏感度。

**項目 6 W7**：照 kvsweep A §8 的 W7-1～W7-9；ICPE 之後再談。

## 10. 第 2 段：慢層上限的 Rust 版（**等使用者同意才做**）

1. 把 `code/p3x/x1fast/` 複製到 `code/icpe/rux_x1fast/v6_slowcap/`，照 3c 的表改；`code/p3x/*` 不動。`x1_rust.py` 的對應改動放在同目錄的複本（只給對拍用）。
2. 語意照 py，一字不改；看不懂就停下來問。
3. 驗證（全部要 0 個不同，不同就停、寫出完整差異）：
   - 功能關閉：G1 式（4,416 列 × 29 欄）＋ xc_sa（32 列 × 40 欄），和 Python 相同。
   - 功能打開：Mooncake conversation R1、SemiAnalysis N16 s1、Bailian coder 的子集。預設子集〔沒意見就照做〕：
     - 慢層 7 種規則全部；CPU 規則 LRU、LRU2、PENS、GDSF、ARC、S3FIFO、LRU+RPF（預取）、LRU+2HA（回合結束丟）、O-BEL、O-NOW；GPU 規則 LRU、BEL、TTLA:10、TTLB:10；CPU 8、111.0625、1,024 GiB；wb 為主，wt 與預取進 GPU 各一格（D1-5）。
     - 雜訊：B1、B2 各至少一組（模型 P 與 M、λ＝0.3、seed 0），配 O-BEL 與慢層 BEL。
     - 逐格比 CSV 每一欄（照 `cmp_noise.py` 的 15 欄鍵加 `slow_cap_frac`、`slow_policy`）；再逐請求比 npz 的每個陣列（`allow_pickle=False`，NaN 視為相同）。
     - Rust 另有驗證模式：每 k 步做 py:1997–2000 的稽核，並斷言 LRU2／PENS／BEL 沒有同分。
4. 速度：同一批子集量 Python 對 Rust（單執行緒、12 執行緒），記 load。
5. 交接：寫 STATUS_RUX，用訊息提醒 icpe-sim；審查、包裝進 `code/p3x/x1fast`、正式 G1 由 icpe-sim 做。

## 11. 錯誤、異常、NOT_MEASURED

- 錯誤：無（這段所有 run exit 0）。
- NOT_MEASURED：TraceLab 的任何時間（D1-6）；硬體計數器（`perf` 沒裝）；Rust 版有上限的時間（還沒有程式）；項目 2–5 新功能本身的時間（沒有程式）；W7 產生器的工作量。
- 時間的限制：三份紀錄同時各一個單執行緒行程、另有 job 形狀的計時同時在跑，load 1.1–3.5／16 執行緒；只拿來做相對比較。SemiAnalysis 只跑 1 次。
- 記憶體：這段所有 run 的 MemAvailable 最低 18,347,532 kB（17.5 GiB），沒有觸發 < 3 GiB 的停止；1 分鐘 load 最高 3.54（每 5 s 一筆，各 run 的 `mem.log`）。
- 摘要 CSV 由 run `20261107-003724-icpe-rux-analysis-csv` 產生：799 列，缺 run_id／ts／host 的 0 列；同一設定各次指紋不同的 0 組；Python 與 Rust v5 指紋不同的 0 組。
- 違規自報：無。暫存檔都在 session 的 scratchpad，沒有寫 `/tmp` 根目錄；Bailian coder 的 npz 寫進 `$ROOT/datasets/x1_nb`（資料目錄，不進 git）。

## 12. run 目錄（`/root/icpe-next/runs/`）

| run_id | 做什麼 | exit |
|:--|:--|:--|
| `20261107-001138-icpe-rux-recheck-noise` | 重跑雜訊 B1／B2 的逐格比對 | 0 |
| `20261107-001533-icpe-rux-convert-bailian-coder` | Bailian coder 轉 npz（最大 RSS 1.25 GB、6 s） | 0 |
| `20261107-001551-icpe-rux-npz-digest-bailian-coder` | Bailian coder 的 npz 指紋（和 acvlab 相同） | 0 |
| `20261107-001705-icpe-rux-prof-sc-smoke` | 驅動程式試跑 | 0 |
| `20261107-001735-icpe-rux-prof-sc-mc` | Mooncake：計時 5 次、Rust v5、cProfile；GPU-BEL 3 次 | 0 |
| `20261107-001737-icpe-rux-prof-sc-bl` | Bailian：計時 3 次、Rust v5、cProfile；GPU-BEL 3 次 | 0 |
| `20261107-001739-icpe-rux-prof-sc-sa` | SemiAnalysis：計時 1 次、Rust v5、cProfile | 0 |
| `20261107-001901-icpe-rux-count-jobs-sc` | `jobs_for` 數 job | 0 |
| `20261107-002132-icpe-rux-first-overflow` | 第一次超過上限的位置 r\* | 0 |
| `20261107-002255-icpe-rux-prof-shapes` | 其他項目最接近的 v5 job 形狀（TTL、CPU 128／256） | 0 |
| `20261107-003724-icpe-rux-analysis-csv` | 產生 `ANALYSIS_rows.csv` | 0 |
