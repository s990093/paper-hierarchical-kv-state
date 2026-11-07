# STATUS_MAIN（icpe-main，acvlab，分支 `icpe-next-main`，worktree `/ssd6/hungwei/icpe-next/main-ana`）

> 只由 `icpe-main` 寫。格式照 NEXT_PLAN §3b：日期時間／做完什麼（commit、run_id）／卡住什麼（完整錯誤）／需要別的 session 做什麼／需要使用者決定什麼。最新的一節在最上面。
> icpe-main＝**正式的主結果分析 session**：算正式的上限表、照 PREREG 下判定。不跑模擬、不動任何佇列、不碰 `/ssd6/hungwei/icpe-next/data`、不改別人的檔（`next_sim_analyze.py` 原樣使用）。自己的輔助腳本＝`code/icpe/main_*.py`。

## 2026-11-07T08:20+08:00 — P4 第二次計算：Bailian、TraceLab 核對全過；**Mooncake 的 λ＝0 自檢失敗**（只在 rule＝min，原因已查明），停下來問使用者

**使用者決定**（本 session，AskUserQuestion，11/7 約 07:5x，看任何 P4 上限數字之前）
- 06:40 節的錯誤：「我先補欄位再餵進去」。`code/icpe/main_addnoisecols.py`（9ec8d94）把 main／`_p1` 的 sweep 逐字複製（只用 csv 模組讀字串，不解析數值），列尾補 `noise_*`＝`load()` 的預設，寫在 run 目錄的 `in/`、不進 git；`next_sim_analyze.py` 原樣使用。
- 3 × 10 合起來的表：「只合湊滿 30 的格」，湊不滿的另列格數。

**run `20261107-080102-icpe-main-p4-noise`**（exit 1；留了 `FAILED_NOT_USED`）
- Bailian、TraceLab：補欄檢查通過；沒有雜訊的上限表和版本 B 逐格相同（mean 與 p90 各 972／972，`out/cellcheck.csv`）；λ＝0 自檢通過（P、M 各 272 格、201 格，mean 與 p90）。
- Mooncake：補欄檢查通過，接著 `next_sim_analyze.py` 停在：
  ```
  File "code/icpe/next_sim_analyze.py", line 188, in main
    assert dmax == 0.0, f"λ＝0 的上限和沒有雜訊的不同（最大差 {dmax}）：{spec}"
  AssertionError: λ＝0 的上限和沒有雜訊的不同（最大差 0.13384690486638032）：('M', 0.0, 'B1', 0)
  ```
- CacheWise、SemiAnalysis 沒有跑到。

**原因**（診斷 run `20261107-081616-icpe-main-p4-diag-l0`；照 `main()` 的雜訊流程重做 λ＝0 的兩個設定，只數不同的格數）
- R1 457、R2 434 個（格, 元件）：**0 個不同**。P、M 都一樣。
- rule＝min 353 個：**31 個不同**，全部是「這格這個元件只有 R1 或只有 R2 有跑雜訊」的；兩個規則都有跑的 322 個：0 個不同。
  - 偏離 8 第 3 點：Mooncake 只列 R1／R2，所以只有一個規則 ≥15% 的格，另一個規則沒有跑雜訊。
  - `analyze()` 的 rule＝min 是對「有的規則」取小，只有一個規則時就等於那個規則，不是真的 min；沒有雜訊時是兩個規則取小。自檢比到這 31 個就不同。
  - 這 31 個的版本 B min 都 <15%（另一個規則 <15%），本來就不在選格清單裡。
- λ＝0 的原始列：P、M 各 11,827 列，和 main／`_p1` 同一個（格, CPU 規則, GPU 規則）的列在 mean_ret、median_ret、p90_ret、n_ret、n_req、sum_ret、rho_w、rho_r、total_all 都逐值相同。
- 〔判讀〕模擬本身在 λ＝0 和沒有雜訊完全相同；失敗來自 `next_sim_analyze.py` 的 rule＝min 遇到「只跑一個規則」的格。

**已看過哪些正式結果**：上面的格數與最大差；Bailian／TraceLab 的核對結果（PASS 與格數）。P4 加雜訊後的上限、seed 間統計，一個都沒看過。

**需要使用者決定什麼**：Mooncake 怎麼處理（下一節）。

## 2026-11-07T06:40+08:00 — P4 資料到齊；第一次計算失敗（`next_sim_analyze.py` 合併沒有雜訊欄的 sweep 時，沒有雜訊的列全部不見）；停下來問使用者

**做完什麼**
- icpe-data 06:40 通知 P4 全部進 git。合併進本分支（只在本機）：
  - f88ca7e＝`origin/icpe-next-data` a25dcba（`p4_b1`／`p4_b2` 68 個 block）。
  - 0812989＝`origin/icpe-next-trx` 3be77ac（`sa_p4_b1`／`sa_p4_b2` 48 個 block）。
  - `PREREG.md` 沒有差異；`next_sim_analyze.py` 三邊都是 blob aa8e3a9。

**卡住什麼**（run `20261107-063333-icpe-main-p4-noise`，exit 1，留了 `FAILED_NOT_USED`）
- 指令：`next_sim_analyze.py --sweep <bl_main> <bl_p1 _ttl> <12 個 bailian P4 檔> --out-prefix bl_p4 --pf-target cpu --metric mean_ret`。第一個紀錄（Bailian）就停了，`out/` 是空的。
- 完整錯誤：
  ```
  File "code/icpe/next_sim_analyze.py", line 177, in main
    hx = H.set_index(ck).headroom
  KeyError: "None of ['trace', 'rule', 'cpu_GiB', 'slow', 'gpu', 'restore', 'gpu_GiB', 'gpu_write', 'slow_cap_frac', 'component'] are in the columns"
  ```
- 原因（診斷 run `20261107-063414-icpe-main-p4-diag`，只數列數、不算上限）：
  - main 和 `_p1` 的 sweep 沒有 `noise_*` 欄，P4 的 sweep 有。
  - `load()` 只在「合併後整欄都沒有」時才補 `noise_model="none"`。合併後這欄存在，所以 main／`_p1` 的 41,472 列變成 NaN，等於 `"none"` 的列是 0。
  - 所以沒有雜訊的上限表 H 是空的，在 λ＝0 自檢之前就停了。
- **已看過哪些正式結果**：和 21:14 節相同。P4 一個上限數字都沒看過（失敗時沒有輸出任何上限）。

**需要使用者決定什麼**：怎麼處理這個錯誤（見下一節）。

## 2026-11-06T21:14+08:00 — P4（W4 雜訊）的分析選擇：使用者在看任何 P4 數字之前決定；全部到齊再算

**使用者決定**（本 session，AskUserQuestion，11/6 約 21:13）
- 當時 P4 只有 SemiAnalysis 進 git：`origin/icpe-next-trx` b6eb4cd，48 個 block、96 個檔。icpe-main 只看了檔名和 CSV 的欄位名稱，**沒有讀任何 P4 的數字**。
- 決定照 PREREG 第 6 項與 §3 B、C；PREREG 沒寫到的部分如下：
  1. **只報告：格數。** 對每種雜訊設定（模型, λ, 抽法），報 10 個 seed 的中位數仍 ≥15%／≥5% 的格數，加上 PREREG 要的 seed 間中位數、最小–最大、平均±標準差。**不下新的判定字**，所以不改 PREREG。
  2. **B1 主、B2 並列。** 主表用 B1，B2 放在旁邊當對照，和 `h200` 在 §3 B 的解讀一致。主模型照 PREREG 是 P。
  3. **佇列 seed：兩種都報，分開為主。** 主表每個佇列 seed（s1／s2／s3）各自 10 個雜訊 seed，和版本 A／B 判定的粒度相同。另報 3 × 10＝30 個合起來的中位數與範圍（§3 C 的「交叉 3 × 10」）。
  4. **全部到齊再算。** 要等 acvlab 的其餘 P4 和 TRX40 的 6 個 Mooncake block 都進 git，SemiAnalysis 不先算。
- 照之前的做法、不另外問的：
  - 上限用 `next_sim_analyze.py`（原樣使用），λ＝0 自檢不過就停。
  - p90 只報告；寫入飽和的格照列，另外標註。
  - 雜訊的 ② 用沒有雜訊的列，也就是 P1 正式的 sweep。

**已看過哪些正式結果**：和 14:23 那節相同。P4 一個數字都沒看過。

**時間**〔推算，別的 session 寫的〕
- acvlab 的其餘 P4 約 11/7 04:40–05:35（STATUS_DATA 19:50，8d70cbc）。
- TRX40 的 6 個 Mooncake block 21:03:52 開始，約 3 h（icpe-data 的訊息、STATUS_TRX 21:15）。
- 全部進 git 後 icpe-data 會通知 icpe-main。

**還沒 push**：照使用者的話，等資料出來再說。本節只在本機。

## 2026-11-06T14:23+08:00 — 第 4 步：P2／P3 敏感度（3 個 TTL）完成；工具類的判定字在每個敏感度都和版本 A 相同

**使用者決定**（本 session；都在 run 開跑 14:16:10 之前、沒有看過任何 P2／P3 結果時）
- 方案二：等 TL 的 `_wt` 進 git，再一次算完。
- 報告形式（AskUserQuestion）：每個敏感度出上限表，並照偏離 7 下判定字。只報告，主判定（版本 A／B）不變。
- 逐格配對（AskUserQuestion）：每組只換一個設定。
  - wt 對 wb；預取進 GPU 對預取到 CPU。
  - 只載入對 Cake，比同一個慢層；nvme807 的只載入對 P2 的 nvme807 Cake。
  - nvme807 Cake 對 ssd Cake（PREREG §2 第 4 項：其他設定和 SSD 級相同）。
- sens_rep 重疊（AskUserQuestion）：先逐格核對；數字只用 P2／P3，不重複算。

**輸入與做法**
- 合併：`origin/icpe-next-data` bb68038 → 0931518；`origin/icpe-next-trx` c8e554f → 2852684。
  - 兩次都沒有衝突；PREREG.md 沒變（對方的版本是我們的位元組前段）。
- 規則集合：17 CPU × 8 GPU（3 個 TTL），和 sens_rep 相同（icpe-data 對偏離 6 的解讀）；對照＝版本 A（`results/icpe_next/main/verA/*_ttl3_*`）。
- 預取進 GPU＝main＋`_pfgpu`，`--pf-target gpu`。`_pfgpu` 的 block 只有 O-PF、LRU+RPF、O-ALL 的列（54 格 × 3 × 8）。
- 工具（b54d3cf）：`main_pair.py`（逐格配對），`main_rowcheck.py` 加 `--expect-value`（每列的設定欄必須是該變體的值）。
- run `20261106-141610-icpe-main-p23-sens`（14:16:10 → 14:21:10，exit 0，stderr 空）；結果 a9b9aa6，`results/icpe_next/main/p23/`（129 個檔＋manifest，12 MB，單檔最大 0.75 MB）。

**核對（全部 PASS）**
- rowcheck 30／30：6 類 × {main、nvme807（18 格）、load（72 格）、wt（54）、pfgpu（54 格 × 3 CPU 規則）}，含每個變體的 slow／restore／gpu_write／pf_target。
- sens_rep 重疊 16／16 IDENTICAL：cw／bl／mc／fi × {nvme807、load、wt、pfgpu}，共 23,200 列。sens_rep 是 Python（f7038ddfaf9c），P2／P3 是 Rust（ed2f4c78c88a+rs43dbe828）。
- 預取進 GPU 時 O-ORD、O-DROP、O-NOW 和版本 A 逐格相同 6／6（這三個元件的 ④、② 都沒有預取規則，必然成立）。
- 格數自檢 8／8；配對 4／4（x 每列都配到 y 的唯一一列）。

**判定（照偏離 7；只報告）**：變體數 ≥15%／全部（判定元件的最大值）

| 類 | 版本 A | nvme807 | 只載入 | wt | 預取進 GPU |
|:--|:--|:--|:--|:--|:--|
| 原始到達、工具類 | 3/9（62.4%） | 2/9（34.2%） | 3/9（99.9%） | 3/9（60.1%） | 3/9（73.7%） |
| 合成到達、工具類 | 21/26（55.0%） | 16/26（52.8%） | 23/26（98.1%） | 22/26（55.1%） | 22/26（54.6%） |
| 原始到達、非工具類（只報告） | 5/5（61.6%） | 4/5（41.2%） | 5/5（99.9%） | 5/5（64.4%） | 5/5（70.4%） |

- 推翻條件：每個敏感度都是「未定：≥15% 成立，但簡單線索（W3）NOT_MEASURED」。
- **nvme807 只有一種慢層（18 格），版本 A 有 ssd／b32／nfs（54 格），判定字不能直接比**。公平的比法是同一格的 ssd：原始工具類 2/9 對 2/9、合成工具類 16 對 17（CacheWise N64 s1：ssd 16.5% → nvme807 14.9%）、原始非工具類 4 對 4。
- 和版本 A 判定字不同的變體（`verAS_verdict.csv`）：
  - 只載入：CacheWise N32 s1、s3 NO → YES（63.2%、57.8%）。
  - wt：CacheWise N32 s1 NO → YES（16.9%）。
  - 預取進 GPU：CacheWise N32 s1 NO → YES（15.6%）。
  - nvme807 的翻轉見上一點（要和 ssd 比）。
- 判定元件 ≥15% 的（格, 元件），不含 rule＝min：版本 A 1,705／10,800（寫入飽和 141）、nvme807 372／3,600（0）、只載入 2,520／14,400（88）、wt 1,689／10,800（130）、預取進 GPU 1,856／10,800（149）。
  - 預取進 GPU 只有 O-PF、O-GPU 會變：O-PF 281 → 467、O-GPU 814 → 779。

**逐格配對**（判定元件、不含 rule＝min；`pair_*_pairsum.csv`）

| 配對 | 對數 | 敏感度較高／較低／相同 | 跨過 15%：往上／往下 |
|:--|--:|:--|:--|
| wt 對 wb | 10,800 | 5,829／4,020／951 | 157／173 |
| 預取進 GPU 對預取到 CPU | 10,800 | 1,303／2,307／7,190 | 233／82 |
| nvme807 對 ssd（Cake） | 3,600 | 389／2,879／332 | 2／61 |
| 只載入對 Cake（同慢層） | 14,400 | 8,823／4,618／959 | 762／319 |

**只載入的極端格**：上限 >0.9 的有 84 個（格, 元件），不含 rule＝min：nfs 的 O-ORD 73、nfs 的 O-PF 6、b32 的 O-ORD 5。
- 這些格 ② 的平均 TTFT 中位數 218 秒、最長 1,300 秒（例：Mooncake toolagent R2、CPU 256 GiB、nfs、GPU 閒：② 684 秒、④ 0.49 秒）；其中 39 格標了寫入飽和。
- 同樣的格在 Cake 下是 10.9%–62.4%。
- 〔判讀〕這是只載入時慢層讀取把對手拖垮的情形，不是「預測能多給多少」的典型值；引用只載入的最大值要附 T 值。

**p90（只報告）** 判定元件最大值：nvme807 42.7%／40.8%／60.1%（原始非工具／原始工具／合成工具）、wt 72.9%／71.7%／87.0%、預取進 GPU 75.5%／83.3%／87.6%、只載入三類都是 1.000（四捨五入）。

**這一節沒做的**：sens_rep 的「快取命中前綴」（cw、tl）和「截斷 128K」（cw、fi、tl、sa）不在我的步驟清單裡；資料在 git，沒有算。

**寫本節時已經看過的正式結果**：前面各節；第 4 步的全部輸出（判定、by_trace、配對、只載入極端格的 T 值、p90 最大值）。W4 雜訊：沒有看過任何結果（還沒跑）。

**需要別的 session 做什麼**：無。

**需要使用者決定什麼**：W3 線索 (a) 要不要列進 ICPE（NEXT_PLAN §7 第 4 項）。沒有 W3，每一版的推翻條件都只能寫「未定」。
- 使用者 11/6（看過本節之後）：「push 跟第二點等數據出來再說」——W3 先不決定；第 4 步的 commit 也先不 push（只在本機）。

**接下來**：P4（W4 雜訊）的結果進 git 後再分析（開算前先問使用者）。

## 2026-11-06T12:42+08:00 — 版本 B（P1 正式值）完成：判定字和版本 A 相同；P4 選格清單；PREREG 偏離 8

**使用者決定**（AskUserQuestion，本 session）：
- P1 進 git 之後（icpe-data 10:46 通知；合併 c472cdb 11:03:54 之前）：版本 B「照計畫開始」；重疊列「要核對」。
- 看過版本 B 全部結果之後（run 11:15:21 跑完之後、baa508d 12:40:25 之前）：「全部 commit 並 push」；P4「加上 O-ALL(report)」；「登記，由 icpe-main 寫」。

**做完什麼**
- 合併：c472cdb＝`origin/icpe-next-data` @ d30ae7c。`PREREG.md` 有衝突，取我這邊：對方那份是我這邊的前段（有偏離 7、沒有偏離 7 更正），所以就是聯集，0 行刪除。df1ec60＝`origin/icpe-next-trx` @ 98b0d2f（sa_p1、sa_p2）。`next_sim_analyze.py` 沒變；`x1_sim.py` 只改了 Rust 的準備流程。
- 核對工具（fb61730）：`main_rowcheck.py`、`main_overlap.py`、`main_ab.py`。P4 選格（50b04e8）：`main_p4cells.py`。
- run `20261106-110735-icpe-main-verB-p1` **失敗**（exit 1），在算任何上限表之前：`FileNotFoundError: 'results/icpe_next/b/sa_main_ttl7/*/x1_sweep.csv.xz'`。原因是我的 cmd.sh 寫錯：`sa_main_ttl7` 只有 `cells.csv`，那次補跑照 STATUS_DATA 08:10 沒跑。run 目錄留了 `FAILED_SCRIPT_BUG`。
- run `20261106-110854-icpe-main-verB-p1`（PASS，6 分 24 秒）只拿掉 SemiAnalysis 那一行重疊核對，其他相同。結果在 `results/icpe_next/main/verB/`、`results/icpe_next/main/oldtests_verB/`（baa508d，已 push）。
  - 正式算法：每類紀錄 main＋`_p1` 的 `_ttl`；④ 和 ② 都有 7 個 TTL；沒有 `--rival-only-gpu`；不含 `cw_main_ttl7`。
  - 並列：偏離 5 的算法（`verB_dev5`）。
  - 只報告：`--dead-h 1800／3600`（加 `_trunc`）、p90。

**核對（全部 PASS）**
- **列數**（`rowcheck.csv`）：18 份 sweep 都是完整網格（main 17×8、`_ttl` 15×8、`_trunc` 6×16，每個（紀錄, 規則）54 格）。42 個（紀錄, 規則）× 每個元件都是 54 格，和版本 A 相同（`verAB_by_trace.csv`）。
- **重疊列**（`overlap.csv`）：`cw_main_ttl7` 對 `cw_p1` 的 `_ttl` 重疊 19,840 列，逐格相同。兩邊程式是 f7038ddfaf9c（Python）對 ed2f4c78c88a+rs43dbe828（Rust）。SemiAnalysis 沒有第二份結果，NOT_MEASURED。
- **逐格關係**（`cellcheck.csv`，30/30）：
  - B 的 O-GPU ≤ A。
  - 偏離 5 的算法 ≤ A。
  - B 和偏離 5 的 O-GPU、O-ALL 完全相同。
  - 截尾版的 O-ORD、O-PF、O-GPU 和不截尾的完全相同。
- **舊模擬器**：`--check-v0` 仍重現論文的 50.5%、95、36。

**判定**（`verB_verdict.csv`；和 A 並列在 `verAB_verdict.csv`）

| 類別 | 版本 A | 版本 B（正式） |
|:--|:--|:--|
| 原始到達・工具類 | 9 個變體中 3 個 ≥15%，最大 62.4% | 3/9，最大 62.4% |
| 合成到達・工具類 | 21/26，最大 55.0% | 21/26，最大 52.9% |
| 原始到達・非工具類（只報告） | 5/5，最大 61.6% | 5/5，最大 61.6% |

- 原始工具類成立的是 Mooncake toolagent R1 57.8%、R2 62.4%，和 TraceLab c1 38.9%。FreeInference 6 週都 ≤0.9%。
- 推翻條件：未定（W3 NOT_MEASURED）。偏離 5 的算法並列：判定字、各類的變體數都相同。

**A 到 B 變了什麼**
- SemiAnalysis 的 O-GPU 降 3–4 個百分點（例：N64 s1 從 54.7% 到 51.7%）；≥15% 的格從 355 降到 333。
- 有些元件因為 ④ 也多了 TTL 而小升：Mooncake toolagent R1 從 57.4% 到 57.8%；原始工具類的 O-PF 最大值從 52.0% 到 53.3%。
- CacheWise 剛好卡在門檻邊：N32 s1 14.8%（不成立），N32 s2 15.4%（成立，2 格）。
- O-NOW 在所有紀錄都沒有一格 ≥15%（最大 8.9%）。
- ≥15% 的 1,694 個（格, 判定元件）裡有 144 個寫入飽和，全部在 Mooncake。

**截尾（只報告，`verB_h1800`／`verB_h3600`）**
- 定義：`-H{H}`＝紀錄最後 H 秒內不准判「不回來」（`x1_sim.py:707`、`:1228`）。
- 只有 O-DROP、O-NOW 會變。O-DROP ≥15% 的格：
  - 原始非工具類 120 → 20（H1800）→ 0（H3600）。
  - 原始工具類 77 → 9 → 0。
  - Mooncake 的 O-DROP 最大值從 55.7% 降到 0.0%（H3600）。
- 〔判讀〕O-DROP 的上限幾乎全部來自紀錄尾端，和 EVIDENCE B2 的舊結論一致；撐住判定的是 O-GPU、O-ORD、O-PF。

**論文的兩個 Mooncake 舊測試**（`oldtests_verB_*`；rule=min、Cake）

| 測試 | old_V0（論文） | old_V1 | new 版本 A | new 版本 B |
|:--|--:|--:|--:|--:|
| T1：conversation O-PF、SSD、CPU 128 GiB | 50.5% | 53.2% | 39.1% | 37.8% |
| T2：192 格 ≥15%（不截尾） | 95 | 97 | 113 | 111 |
| T2：不截尾、只算不飽和 | 62 | 65 | 79 | 77 |
| T2：截尾 H3600＋只算不飽和 | 36 | 38 | NOT_MEASURED | **38** |

- 新 B 的 38 格依 CPU 分：128 GiB 16、256 GiB 15、512 GiB 7、1,024 GiB 0（V0 是 16／12／8／0）。
- 依慢層分：ssd 17、b32 21、nfs 0（V0 是 19／17／0）。依元件分：O-ORD 18、O-PF 20、O-DROP 0。
- 只報告：R1 40 格、R2 42 格；O-GPU 25 格。

**P4（W4 雜訊）選格清單：給 icpe-data 排模擬**（PREREG 偏離 8；run `20261106-124122-icpe-main-p4cells`；43134be）
- 選法：版本 B 正式上限 ≥0.15 的每一（格, 元件）。元件＝O-ORD、O-DROP、O-NOW、O-PF、O-GPU＋O-ALL(report)；O-ALL 加雜訊後仍只報告。Mooncake 只列 R1／R2（min 已涵蓋，程式自檢 322 筆）。
- 規模：2,582 個（格, 元件）、975 個不同的格。
  - 依元件：O-ALL(report) 888、O-GPU 781、O-ORD 434、O-PF 282、O-DROP 197、O-NOW 0。
  - 依紀錄（格×元件／格）：Mooncake 891／216、SemiAnalysis 849／389、CacheWise 369／179、Bailian 272／98、TraceLab 201／93、FreeInference 0。
- 檔案（`results/icpe_next/main/p4/`）：
  - `p4_cells_distinct.csv`：每格一列，`components`＝要加雜訊的元件；排模擬用這份。
  - `p4_cells.csv.xz`：每個（格, 元件）一列，含 ④／② 的做法、headroom、`write_saturated`。
  - `p4_summary.csv`：格數。
  - 格的座標＝(trace, rule, cpu_GiB, slow, gpu, restore, gpu_GiB, gpu_write, slow_cap_frac)，全部是主時間設定（wb、111.0625 GiB、cake）。
- 雜訊的模型、抽法、λ、seed、λ＝0 自檢，照 §3 B／偏離 6，偏離 8 沒有改。

**PREREG 偏離 8**（df8b74c，append-only，15 行新增、0 行刪除）：P4 選格的解讀。號碼由 h200 確認可用（h200 之後從 9 開始編）。

**寫本節時已經看過的正式結果**：版本 A 全部（前面各節）；版本 B 全部輸出（判定、偏離 5 的算法、截尾、p90、舊測試、各元件格數）；P4 清單。W4 雜訊：沒有看過任何結果（還沒跑）。

**需要別的 session 做什麼**
- icpe-data：依 `results/icpe_next/main/p4/p4_cells_distinct.csv` 排 P4。
- h200：同步 PREREG 偏離 8（df8b74c）。

**接下來**：第 4 步 P2／P3，等 `_p2`／`_p3` 進 git，在 3 個 TTL 的基礎上算，不重複算和 `sens_rep` 重疊的代表格。目前 git 上：sa_p2 已完成；sa_p3 的第一個 block 失敗（STATUS_TRX：Rust panic），等 icpe-trx 處理。

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
