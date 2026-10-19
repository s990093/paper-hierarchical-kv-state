# X1　oracle 拆解：猜對「誰回來」值多少、值在哪一種決定上？〔模擬，CPU〕

（§2 在寫任何策略模擬程式之前寫死；§0、§1、§3–§7 是結果出來後才寫的。）

## 0. 一句話結論＋判定

**判定（原登記版：只用 Mooncake、每格取 R1／R2 兩種重建規則中較小的 headroom）：有看頭**〔模擬，MI300X 常數；模擬只能篩，不能進論文〕。事後的截尾穩健版（「之後沒人讀」只用在離 trace 結束 ≥H 秒的決定，H＝1,800／3,600 s）**也是有看頭**，但成立的元件變了：**O-DROP 的空間大半是 trace 截尾造成的**（H＝1,800 s 時本地 SSD 上只剩 3.6–12.1％；H＝3,600 s 時等於 LRU），能留下來的是 **O-PF（預取）與 O-ORD（降級順序）**，它們不用「不回來」這個 bit，不受這個檢查影響。總結以較保守的版本為準：**有看頭，只靠 O-PF 與 O-ORD**。

- **兩份 Mooncake trace 不獨立**：conversation 的 12,031 個請求中有 10,938 個在 toolagent 有相同的（timestamp、輸入長度、輸出長度），block 數也相同〔資料，run 20261017-221242-p3x1-posthoc〕。它們只能算一份證據。**判定只用 conversation 也成立**（例如本地 SSD、GPU 閒：O-PF 在 128／256／512 GiB 是 44.5／40.2／24.6％、O-ORD 22.5／24.1／16.7％）；toolagent 只當同一份資料的另一種切法。

- **值在哪裡**：原登記版三個元件都過門檻，**預取（O-PF）最大、降級順序（O-ORD）次之、回合結束就丟（O-DROP）第三**（但 O-DROP 過不了截尾檢查，見上）；「不回來的不寫慢層」（O-NOW）在所有格子都 <5%，而且在 NFS 上比「完全不寫」還慢——這一項沒有價值。
- **和 X5 不同的地方**：X5 說幾乎全部價值來自「知道誰不會再回來」。我這裡在降級那一側看到同樣的現象（O-DROP 只用 1 個 bit，在本地 SSD 上拿到 Belady 改善的 77–105％），**但這一部分正是截尾檢查拿掉的**；留下來的是需要「何時」的預取，以及不靠「丟掉」的 Belady 順序。可能的原因〔判讀〕：(1) X5 沒有預取這個動作；(2) X5 的「不會回來」如果也是用 58.9 分鐘的 trace 定義，就有同樣的截尾偏差；(3) Belady 只是降級、不丟，trace 結束後才回來的 session 在真實的未來裡也一樣排在最後，所以順序本身不受截尾影響，但「丟掉」會讓它們之後要整段重算，這個代價 trace 量不到。
- **不靠寫入飽和的格子也過**：本地 SSD（ρ_w≈0.5）、GPU 閒、CPU 128／256 GiB（工作集的 1.2％／2.4％），conversation 的 headroom：O-PF 44.5％／40.2％、O-DROP 22.8％／28.2％、O-ORD 22.5％／24.1％；toolagent：25.4％／20.2％、17.2％／15.6％、11.7％／8.8％〔run 20261017-180517-p3x1-sweep → 20261017-182005-p3x1-analyze，`x1_headroom_components.csv` rule=min〕。session-cluster bootstrap：headroom ≥15% 的 88 格裡 79 格的 95% CI 下界也 ≥15%〔run 20261017-213626-p3x1-boot〕。
- **空間隨容量很快收掉**（容量也換成「CPU 裝得下幾分鐘的歷史」＝LRU 下 session 從回合結束到被降級的平均時間：128／256／512／1024 GiB＝0.46／0.93／1.96／4.30 分鐘〔conversation，run 20261017-215113-p3x1-postsweep〕，全部落在 X2 說的「只裝得下最近 1–3 分鐘」區域附近）：CPU 1 TiB（工作集 9.6％）時，本地 SSD 上三個元件都只剩 5–11％（conversation）、−1.1–1.5％（toolagent）。
- **Cake 只縮小一部分**：只載入 headroom >1% 的 142 格裡，Cake÷只載入 ≤0.5 的只有 27 格（19％）；各元件比值的中位數 0.62–0.77（Cake 讓空間小 25–40％，不到一半）→ 依 §2.7 的報告規則，**不成立「Cake 縮小預測的價值」**。
- **FreeInference（事後加入、另列）：死路**。前 6 週、128,072 個回來請求，四個元件 headroom 最大 0.22％；LRU 自己就在 oracle 的 0.3％內——這份 trace 的同時活躍 session 太少，128 GiB 都沒有壓力（ρ_w≈0）〔run 20261017-181951-p3x1-fisweep → 20261017-213544-p3x1-analyze-fi〕。含 FreeInference 的判定仍是有看頭（來自 Mooncake）。
- **oracle 不一定是上界**：175 個「oracle 元件比某個對稱對手慢」的格子，幾乎全在 NFS（O-NOW、O-DROP 輸給「完全不寫」或 2-hit：NFS 上寫入本身有害，而 oracle 只知道誰回來、不管成本）；O-ORD／O-PF 只有 7 格、最多慢 1.1％（toolagent、1 TiB、本地 SSD）。
- **最強的非 oracle 對手很便宜**：小容量時是 LRU-2（session 版 LRU-K），NFS 上是「第 1 輪就丟」的 2-hit 准入（LRU+2HA，比 LRU 快 37–44％）。oracle 的空間是扣掉這些之後剩下的。事後加的 session 版 2Q（先比請求數、再比閒置；X2 提醒）在 96 格都不是最好的對手（比原登記最好的順序對手慢 2–45％），headroom 不變〔run 20261017-221242-p3x1-posthoc〕。

## 1. 問題與要證明什麼

- **問題**：論文的設計是「BF16 全部進 CPU、滿了才降級、降級時壓成 INT4、回來時 Cake 決定重算多少」。延後之後還剩三種決定要猜未來：**降級誰**、**丟掉誰**、**先把誰搬回 CPU**（預取）；另外一個是「不回來的要不要寫慢層」。把「完美知道未來」拆成四個元件，一次只開一個，各自對上「機制完全相同、只差資訊」的最好非 oracle 對手：
  - O-ORD（降級順序：Belady）、O-DROP（不回來的回合結束就丟）、O-NOW（不回來的不寫慢層）、O-PF（Aggressive 預取，受慢層讀頻寬限制、和需求讀取搶同一條線）。
- **要證明什麼**：哪一個元件、在哪些格子（容量、慢層、GPU 忙閒）讓「回來的請求」的完整 TTFT 平均快 ≥15%（有看頭），還是全部 <5%（死路）。順便回答：Cake 會不會縮小這個空間（Cake ÷ 只載入的 headroom 比值），以及 oracle 是不是真的不慢於對手。
- **和 D4 的差別**（D4 已回答、這裡不重做的）：D4 是 block 層級、主指標是全部請求的 TTFT 總和（95% 是所有策略都一樣的第一次重算），沒有預取、沒有慢層 INT4、沒有寫入時間線。這裡改成 session 層級的決定、只算回來的請求、加預取與慢層 INT4、讀寫共用 I/O 時間線（第一階段的 share 模型）。
- **這一輪不回答「猜不猜得到」**（那是 X2、X3）。這裡只量「猜對了值多少、值在哪一種決定」。

## 2. 判準（開跑前寫死）

**寫於**：2026-10-17T09:52:26Z（`date -u`），repo HEAD `75ce8a1`。寫的時候只跑過 trace 載入與 session 重建驗算（run `20261017-175203-p3x1-char`，`results/p3_explore/x1/x1_trace_char.csv`），**沒有寫、也沒有跑任何策略模擬**。

### 2.1 工作負載與 session 重建

- trace：Mooncake FAST'25 `conversation`（12,031 請求）、`toolagent`（23,608 請求），原始長度、原始 timestamp（毫秒；規則 6 的三條斷言在 `code/p3x/x1_trace.py::load_trace`、`unit_check_time`）。
- 每個請求只存**完整 block**（`floor(input_length/512)` 個；最後不滿的 block 下一輪內容一定會變，不存）。最後不滿的 block 算在新一輪 prefill。
- session 重建（`x1_trace.reconstruct`）：r 的 hash 從後往前（k ≥ 1）找最近一個「以這個 hash 為最後完整 block」的請求 p；p 還沒被接續、且 `input(r) ≥ input(p)+output(p)` → r 接續 p 的 session（**回來的請求**）；否則 r 是分叉或新 session。
- **共用前綴：用 refcount（不用「每個 session 自己一份」）**。理由：conversation 的 block 0 被全部請求共用、toolagent 有 4 種 block 0（最大的占 46%），而且「跨 session 的重用存取」在 toolagent 占全部完整 block 重用的 47%（107,361／226,058，run 20261017-175203-p3x1-char）。各自一份會把同一個 system prompt 算成上萬份、也拿不到別人放在 CPU 的前綴；refcount 和 vLLM／SGLang 的前綴快取一致。規則：
  - 一個 block 只有一份實體位置。它在 CPU ⇔ 至少一個「狀態＝CPU」的 session 持有它（或它是預取進來的）；在慢層 ⇔ 寫過慢層且還有活著的 session 持有它；沒有活著的 session 持有 → 釋放。
  - **決策單位是 session**：降級一個 session 只移走「只有它在 CPU 持有」的 block（前綴鏈保證這是一段後綴）；丟掉一個 session 只釋放「只有它持有」的 block。
  - 一個 session 在 CPU 持有的 block 都被別的 CPU session 共用時，降級它不會空出空間，就繼續選下一個。

### 2.2 系統模型（延後設計）

- 時鐘：trace 原始時間，依 timestamp 一次處理一個請求，**GPU 不排隊**（和 D4 相同）。CPU 層與慢層各有一條 I/O 時間線（讀、寫各自排隊；寫入進行中讀取變慢 k_read 倍＝第一階段的 `share` 模型，`m7_restore_harness.Tier` 同公式）；還原、降級寫入、預取都在這兩條線上搶。
- 回合結束（到達時間＋TTFT）：session 的完整 block 全部以 BF16 放進 CPU（新的 block 走 CPU 寫入線）。
- CPU 滿（已用位元組 > 容量）：依策略選一個 session，**降級**（只寫「還沒有慢層副本」的 block，寫 INT4 KIVI 20 MiB／block）或**丟掉**。降級採第一階段的 `free` 語意：CPU 空間立即釋放、位置立即算在慢層、寫入佔用慢層寫入線。慢層容量不設上限。
- 回來時：layout＝前綴 [0,L) 有存（CPU BF16、CPU INT4（預取的）、慢層 INT4），之後沒存。還原＝**Cake** 在 [0,L) 上的兩條線（GPU 從前面重算 f(i)、I/O 從後面載入；`m7_sim.sim_restore` 同演算法），接著重算 [L,n)（含新一輪 prefill）。**只載入**＝[0,L) 依序全部載入，再重算其餘。
- INT4 block 的載入時間＝位元組 ÷ 頻寬（寫入進行中 × k_read）＋反量化 d（GPU 上、放在載入線上；`m10_iowall.load_model` 的模型，GPU 驗過中位誤差 1.1%）。
- 預取（只有預取類策略）：慢層讀取線空著時，把某個慢層 session 的 block 由後往前讀進 CPU（以 INT4 放，20 MiB），讀到「全部在 CPU 時的會合點」為止（前段反正會被 Cake 重算）；和需求讀取共用同一條讀取線（需求讀取到達時，已開始的那個 block 讀完才輪到它，之後預取暫停到需求讀取排完）；讀完前不能用（需求讀取要等它到）。預取進來的 block 釘在 CPU 直到該 session 回來；CPU 滿到只剩預取的 block 時，最後手段是丟掉預取的 block（oracle：回來最晚的先丟；近期版：最舊的先丟）。

### 2.3 成本常數〔MI300X 常數〕（不從 trace 估任何東西）

| 常數 | 值 | 來源 |
|:--|:--|:--|
| f(i) 重算（GPU 閒） | C1 每個 chunk 的中位數，i ≥ 80 線性外插 | `results/m7_write_policy_mi300x/calib_c1.csv`（run 20261008-130316-m7-c1），`m8_trace_oracle.load_f` |
| GPU 忙 | f × 2.07 | `busy_levels.json`（runs 20261008-130834-m7-c5、20261008-130956-m7-c5b） |
| CPU 層 | 35.4159 GiB/s 讀寫、k_read 1.161；BF16 block 64 MiB | `tier_params.json` |
| 慢層：本地 SSD | 讀 6.98115、寫 1.91065 GiB/s、k_read 3.306 | `tier_params.json` local |
| 慢層：b\*＝32 | 讀 1.4668 GiB/s；**寫＝讀、k_read＝1**（限速層沒有量過寫入與干擾，這是假設） | `results/m10_precision_mi300x/speed_llama/bw.json` |
| 慢層：NFS | 讀 0.331084、寫 0.600996 GiB/s、k_read 1.673 | `tier_params.json` nfs |
| INT4 KIVI block | 20,971,520 B | `results/m10_tier_mi300x/iowall.json` bytes_per_block |
| INT4 反量化 d | 閒 0.447 ms、忙 1.701 ms／block | run 20261016-034902-m10-grid-int4-llama，E1 load_only cpu 的 t_dequant ÷ 讀到的低精度 block 數，中位數（`m10_iowall.dequant_per_block` 同法） |

### 2.4 策略（參數一律寫死成文獻或既有程式的預設值）

基底機制＝延後設計（上一節）。非 oracle 策略只用「已經發生的事」；oracle 用 trace 的未來。

| 代號 | 類別 | 做什麼 | 參數 |
|:--|:--|:--|:--|
| LRU | 順序 | 降級最久沒來的 session | — |
| LRU2 | 順序 | LRU-K（K＝2，O'Neil 等 SIGMOD'93）：降級「倒數第 2 次到達」最舊的；只來過 1 次的最先、之間用 LRU | K＝2，無 CRP |
| ARC | 順序 | ARC（Megiddo & Modha FAST'03）的 session 版，T1/T2/B1/B2 與 p 以位元組計 | 自調，無參數 |
| S3FIFO | 順序 | S3-FIFO（Yang 等 SOSP'23）的 session 版，位元組計 | small＝10%、freq 上限 3、S→M 門檻＝在 S 裡被存取 ≥1 次（同 `m8_trace_oracle.run_s3fifo`） |
| GDSF | 順序 | GreedyDual-Size-Frequency（Cherkasova 1998）：H＝L＋freq×cost÷size，降 H 最小的；cost＝Σ(min(f(i), 慢層載入) − CPU 載入)（同 D4 cost-Belady 的權重），size＝會空出的位元組 | — |
| PENS | 順序 | Pensieve 式：V＝Σ f(i)（會空出的 block）÷ 閒置秒數，降 V 最小的 | — |
| LRU+NONE | 丟 | LRU 順序，被選中的一律丟（不寫慢層） | — |
| LRU+2HW | 丟 | LRU 順序；被選中時，這個 session 到目前來過 ≥2 次才寫慢層，否則丟（SGLang `write_through_selective`／KVBM frequency ≥2 的 session 版） | 門檻 2 |
| LRU+2HA | 丟 | 2-hit 准入：session 第 1 次來的回合結束就丟（不留 CPU），第 2 次以後才留；之後 LRU | 門檻 2 |
| LRU+RPF | 預取 | LRU；預取機制同 O-PF，候選＝最近被用過的慢層 session；騰空間只准降級「上次存取比候選更舊」的 CPU session | — |
| PRED(recency) | 檢查用 | 「通用預測器介面」：降級預測下次到達最晚的；預測＝上次存取的請求序號（越舊越晚） | 不變量用 |
| **O-ORD** | oracle | 降級順序用未來：Belady（下次被讀最晚）與 Belady-Size（下次被讀的距離 × 會空出的位元組；LRB NSDI'20 的離線上限）兩個，每格取較好的（B7 教訓：oracle 要取一族最好） | — |
| **O-DROP** | oracle | 回合結束時，如果這個 session「只有它持有的 block」之後再也沒人讀，就直接丟；其餘 LRU、降級照寫 | — |
| **O-NOW** | oracle | LRU 順序；被選中時，之後再也沒人讀的就丟、不寫慢層 | — |
| **O-PF** | oracle | LRU 順序＋Aggressive 預取（Cao、Felten、Karlin、Li，SIGMETRICS'95）：讀取線空著時，預取「下次被讀最早」的慢層 session；CPU 不夠時，只准降級「下次被讀比它更晚」的 CPU session（最晚的先） | — |
| O-ALL | oracle（只報告） | O-ORD（Belady）＋O-DROP＋O-NOW＋O-PF 全開 | — |

「下次被讀」的定義：session 只有它持有（降級時看 CPU、丟掉時看全部）的第一個 block，在目前請求之後第一次被任何請求讀到的時間。

**對稱對手（事先寫死；每格取回來請求平均 TTFT 最低的那個）**：

| oracle 元件 | 機制 | 對稱對手集合 |
|:--|:--|:--|
| O-ORD | 只改降級順序（一律寫慢層、不丟、不預取） | LRU、LRU2、ARC、S3FIFO、GDSF、PENS |
| O-DROP | LRU 順序＋可以丟 | LRU、LRU+NONE、LRU+2HW、LRU+2HA |
| O-NOW | LRU 順序＋可以不寫慢層 | LRU、LRU+NONE、LRU+2HW |
| O-PF | LRU 順序＋預取 | LRU（不預取＝Cake 原樣）、LRU+RPF |

沒有放的對手與理由：TTL 類（Continuum）需要工具時長或一個 TTL 值，trace 沒有工具名，文獻也沒有可直接用的預設 → 不放（§6 列為缺口）。WA 查表要從資料估 → CLAUDE.md §8 禁止。CachedAttention 的「佇列看得到才預取」在不排隊的模型裡提前量是 0，等於不預取（＝LRU）。

### 2.5 格子

trace（conversation、toolagent）× CPU 容量 128／256／512／1024 GiB（同時報占工作集的百分比；工作集＝不同完整 block × 64 MiB＝10,681 GiB／9,999 GiB）× 慢層（本地 SSD、b\*32、NFS）× GPU（閒、忙）× 還原（Cake、只載入）＝ 96 格／策略。長度只用原始長度（「映射到 8K–32K」本輪不做，§6 列）。

### 2.6 指標

- **主指標**：回來的請求（重建規則的 cont）的完整 TTFT 平均值（含新一輪 prefill）。
- 一起報：中位數、p90、「還原舊 context」部分（到 [0,max(L, 上一輪完整 block 數)) 都就緒的時間）的平均、全部請求 TTFT 總和、慢層寫入 GiB、慢層寫入負載 ρ_w＝寫入忙碌時間 ÷ trace 長度、預取 GiB、CPU／慢層／重算 block 數。
- headroom（每個元件、每格）＝(T_最好對稱對手 − T_oracle 元件) ÷ T_最好對稱對手，T＝主指標。

### 2.7 判定（照 explore/README §2 X1，字面套用）

- **有看頭**：某個元件，在至少一個 trace、同一個（慢層, GPU 狀態）下、**Cake 還原**、至少 2 種 CPU 大小，headroom ≥ 15%。
- **死路**：四個元件在**所有** 96 格（含只載入）headroom 都 < 5%。
- 其他＝**可能**。
- 和 README 的差別（只會更嚴，不會更鬆）：「有看頭」只算 Cake（論文的設計是 Cake；只載入格只拿來算比值）；「死路」把只載入格也算進去。
- **寫入飽和標記**：任一格若 oracle 或其最好對手的 ρ_w ≥ 1（寫不完、`free` 語意讓還沒寫完的資料就能讀，物理上不成立），標 `write_saturated`。判定照算，但如果「有看頭」只靠飽和格成立，報告時要明寫「靠 free 語意」。
- 只報告、不改判定：
  - Cake 對只載入的 headroom 比值＝headroom(Cake) ÷ headroom(只載入)（只算只載入 headroom > 1% 的格）；≤ 0.5 的格占多數 ⇒「Cake 縮小預測的價值」。
  - oracle 元件是否每格都不慢於對稱對手；違反的格逐一列出。
  - O-ALL 對所有非 oracle 策略中最好的那個。

### 2.8 不變量（每次 sweep 前自動跑，任一條不過就停）

1. PRED(recency) 和 LRU 的每個請求 TTFT（全部 12 種時間設定）**逐位元相同**。
2. 自己寫的 Cake 還原（BF16、無反量化）和 `m7_sim.sim_restore` 在隨機 layout 上逐位元相同；自己寫的 I/O 線和 `m7_restore_harness.Tier`（share 模式）逐位元相同。
3. 多時間設定一次跑（非預取策略的狀態演變和時間無關）和逐一跑，結果逐位元相同。
4. refcount 守恆：每一步之後重算 CPU 已用位元組，和記帳值相同；容量執行完後已用 ≤ 容量（只剩預取 block 的情況照 §2.2 的最後手段）。

### 2.9 事後補充（照 explore/README §4，登記之後才加；照實記下加的時候看過什麼）

- **重建規則 R2（2026-10-17T10:05Z 加）**：README §4-2 要求 Mooncake 至少報兩種重建規則、判定照 headroom 較小的那個。R2＝R1 再加兩個條件：(a) 整份 trace 裡以 p 為最深候選父的請求只有 r 一個；(b) p 的最後一個完整 block 不是任何其他請求的最後一個完整 block。不符合就當分叉。**正式判定＝每格取 R1、R2 兩者中較小的 headroom**。沒有用文獻 agent 的「前綴被 ≤3 個請求共用」：一個有 k 輪的 session，第 1 輪的前綴會被後面 k−1 輪共用，那條規則會把 4 輪以上的 session 的前幾個連結全部拆掉（只留短 session）。
  - 加 R2 時已經看過的結果：一次冒煙（scratchpad，不進 results）conversation／R1／CPU 256 GiB／NFS／GPU 閒／Cake 一格的回來平均 TTFT：LRU 1.89 s、ARC 1.82、S3FIFO 1.86、GDSF 1.66、PENS 1.83、LRU+2HA 1.06、O-BEL 0.54、O-DROP 0.80、O-PF 1.41；以及 R1 的 trace 特性（`x1_trace_char.csv`）。R2 的定義沒有依這些數字調。
- **FreeInference（2026-10-17T10:08Z 加）**：README §4-1 的新 trace，登記之後才加，判定列兩版（不含／含），總結以不含版為準。範圍事先固定成：週依日曆順序取到累計 session ≥ 1,000 為止＝前 6 週（2026-05-17 到 06-27，1,645 個 session、129,717 個請求；每週 session 數來自 `sessions.parquet` 的列數）；**只跑 Cake**（算力；只載入格不做）；其餘格子與策略同 §2.4–2.5。每週獨立重播（session 不跨週），回來請求平均＝各週總和 ÷ 各週個數。換算：32 個 16-token block 併成一個 512-token chunk（key＝第 32 個 block 的鏈式 id）；o200k_base 的 512 token 直接當成 Llama 的 512 token（比值 NOT_MEASURED）。「回來」＝同一個 session_id 的第 2 個以後的請求（工具觸發也算；使用者發起的另外報）。
  - 加的時候已經看過的結果：上面那一格冒煙；FreeInference 第 1 週一次冒煙（CPU 256 GiB／NFS／閒／Cake：LRU 與 O-BEL 回來平均都是 2.583 s，幾乎全部在 CPU 命中）；主 sweep 進度列印的少數 O-ALL 數字（CPU 128 GiB）。

## 3. 做了什麼

所有指令都走 `TIARA_NO_GPU=1 bash code/run3090.sh <短名> /ssd8/hungwei/paper-hkv/venv/tiara/bin/python -I code/p3x/<程式> ...`（CLAUDE.md §4.1 的殼；run 目錄在 `/ssd8/hungwei/paper-hkv/runs/<run_id>/`，各有 cmd.sh、context.txt、stdout.log、stderr.log、exit_code）。只用 CPU，同時最多 8 個 process。

| 步驟 | 指令 | run_id | 產出 |
|:--|:--|:--|:--|
| 載入驗證、重建、Bailian 對拍 | `x1_trace.py char` | 20261017-175203-p3x1-char | `x1_trace_char.csv` |
| 兩種重建規則的特性 | `x1_trace.py char --bailian` | 20261017-180326-p3x1-char2 | 同上（追加 R1／R2 列） |
| 不變量（§2.8） | `x1_sim.py selftest` | 20261017-180344-p3x1-selftest | `x1_invariants.csv`（44 條全過） |
| **主 sweep（原登記）** | `x1_sim.py sweep --workers 8` | **20261017-180517-p3x1-sweep**（code_sha e59ec2482741；程式快照在 run 目錄 `code_snapshot/`） | `x1_sweep.csv`（3,072 列）、逐請求 TTFT `req_*.npz` |
| **主判定（原登記）** | `x1_sim.py analyze --sweep x1_sweep.csv --traces conversation toolagent` | **20261017-182005-p3x1-analyze** | `x1_headroom.csv`、`x1_headroom_components.csv`、`x1_verdict.csv`、`x1_hot_groups.csv`、`x1_cake_vs_load.csv`、`x1_violations.csv` |
| FreeInference sweep（事後） | `x1_sim.py sweep --workers 7 --traces fi:<前 6 週> --rules session_id --modes cake --out x1_sweep_fi.csv` | 20261017-181951-p3x1-fisweep | `x1_sweep_fi.csv`（2,304 列） |
| FreeInference 判定（事後） | `analyze --sweep x1_sweep_fi.csv --traces freeinference --out-prefix x1fi`（另一個 `--metric mean_ret_user`） | 20261017-213544-p3x1-analyze-fi、20261017-213548-p3x1-analyze-fiuser | `x1fi_*.csv`、`x1fiuser_*.csv` |
| 含 FreeInference 的判定 | `analyze --sweep x1_sweep.csv x1_sweep_fi.csv --traces conversation toolagent freeinference --out-prefix x1withfi` | 20261017-213355-p3x1-analyze-all | `x1withfi_*.csv` |
| 相對 LRU 的改善表 | `x1_report.py --sweep x1_sweep.csv x1_sweep_fi.csv` | 20261017-213401-p3x1-report | `x1_value_vs_lru.csv` |
| session-cluster bootstrap | `x1_boot.py --run-dirs <主 sweep 的 run 目錄> --B 2000` | 20261017-213626-p3x1-boot | `x1_bootstrap.csv` |
| 穩健性：截尾（事後，X9 提醒） | `sweep --workers 7 --policies O-DROP-H1800 O-DROP-H3600 O-NOW-H1800 O-NOW-H3600 O-ALL-H1800 O-ALL-H3600 --out x1_sweep_h.csv` | 20261017-213516-p3x1-hsweep | `x1_sweep_h.csv` |
| 事後對手 FREQ2Q＋LRU 停留時間 | `sweep --workers 1 --policies FREQ2Q LRU --out x1_sweep_post.csv` | 20261017-215113-p3x1-postsweep | `x1_sweep_post.csv`（LRU 和主 sweep 192 格逐格相同） |
| 事後檢查：截尾判定、FREQ2Q、停留時間、trace 重疊 | `x1_post.py all` | 20261017-221242-p3x1-posthoc | `x1_posthoc_checks.csv` |

程式：`code/p3x/x1_trace.py`（載入、斷言、重建）、`x1_sim.py`（模擬器、不變量、sweep、判定）、`x1_fi.py`（FreeInference 載入與換算）、`x1_report.py`、`x1_boot.py`。都只 import 既有的 `m8_trace_oracle.load_f`、`m7_sim.sim_restore`、`m7_restore_harness.Tier`，沒有改既有檔。

## 4. 結果

標記：〔模擬〕＝本模擬器跑出來的數字（MI300X 常數，不是 GPU 量測）；〔資料〕＝trace 本身算出來的；〔判讀〕＝我的解讀。

### 4.1 trace 與重建（規則 6）

| 項目 | conversation | toolagent | 來源 |
|:--|:--|:--|:--|
| `len(hash_ids)==ceil(input/512)`、前綴鏈違反 | 100％、0 | 100％、0 | 〔資料〕run 20261017-175203-p3x1-char |
| 同 session 間隔 ÷ 上一輪輸出 token（中位，ms 解讀） | 347 | 346 | 同上（D4 是 344／339） |
| 回來的請求（R1／R2） | 3,511／2,357 | 7,132／2,378 | 〔資料〕run 20261017-180326-p3x1-char2 |
| 有回來的 session（R1／R2；門檻 1,000） | 1,923／1,581 | 4,862／1,631 | 同上 |
| 回來間隔中位（R1／R2） | 129 s／147 s | 3 s／141 s（R1 有 43％ 是 0 s） | 同上 |
| 工作集（不重複完整 block × 64 MiB，refcount 的分母） | 10,681 GiB | 9,999 GiB | 同上 |
| 跨 session 的重用存取（占完整 block 重用） | 23,962／105,592（23％） | 107,361／226,058（47％） | run 20261017-175203-p3x1-char |

- 重建規則對拍 Bailian（自帶 parent_chat_id，16-token block，同一套前綴鏈規則 R1）：traceA precision 96.8％、recall 98.4％；coder precision 79.8％、recall 98.3％〔資料，run 20261017-175203-p3x1-char〕。
- toolagent 在 R1 下有 43％ 的「回來」間隔是 0 s（同一秒內、同一個前綴的大量請求，〔判讀〕多半是 agent 平行展開）；R2 把這些全部拆成分叉，回來間隔中位變 141 s。所以 toolagent 的兩種規則差很多，判定取較小的那個很重要。
- CPU 容量占工作集：128／256／512／1024 GiB＝conversation 1.2／2.4／4.8／9.6％、toolagent 1.3／2.6／5.1／10.2％（分母＝refcount 下不重複的完整 block）。

### 4.2 主判定（原登記版）：每格取 R1／R2 中較小的 headroom，Cake

headroom（％）＝(最好對稱對手 − oracle 元件) ÷ 最好對稱對手，回來請求的平均完整 TTFT〔模擬，run 20261017-180517-p3x1-sweep → 20261017-182005-p3x1-analyze，`x1_headroom_components.csv` rule=min〕。LRU 的寫入負載 ρ_w：NFS 1.34–1.56（寫入飽和）、本地 SSD 0.42–0.49、b\*32 0.55–0.64（GPU 閒，`x1_value_vs_lru.csv`）。

| trace | 慢層 | GPU | 元件 | 128 GiB | 256 | 512 | 1024 |
|:--|:--|:--|:--|--:|--:|--:|--:|
| conversation | 本地 SSD | 閒 | O-PF | **44.5** | **40.2** | **24.6** | 10.9 |
| | | | O-DROP | **22.8** | **28.2** | **18.6** | 7.0 |
| | | | O-ORD | **22.5** | **24.1** | **16.7** | 5.3 |
| | | | O-NOW | 3.9 | 3.1 | 2.9 | 2.6 |
| | | 忙 | O-PF／O-DROP／O-ORD | 30.9／14.7／18.5 | 26.9／20.6／19.8 | 15.2／14.4／13.6 | 5.9／6.2／5.1 |
| | b\*32 | 閒 | O-PF／O-DROP／O-ORD | 52.0／15.2／27.1 | 47.7／27.0／30.6 | 31.6／22.1／24.0 | 15.8／11.7／10.1 |
| | NFS | 閒 | O-PF／O-DROP／O-ORD／O-NOW | 9.1／−17.6／49.3／−44.1 | 24.8／20.8／62.0／−36.5 | 40.2／45.2／63.7／−13.5 | 47.2／43.2／43.6／3.6 |
| toolagent | 本地 SSD | 閒 | O-PF／O-DROP／O-ORD | **25.4**／**17.2**／11.7 | **20.2**／**15.6**／8.8 | 8.1／3.4／2.0 | 1.5／−0.8／−1.1 |
| | b\*32 | 閒 | O-PF／O-DROP／O-ORD | 34.0／11.9／17.0 | 28.1／18.0／15.5 | 13.5／8.7／6.9 | 3.8／1.6／0.9 |
| | NFS | 閒 | O-PF／O-DROP／O-ORD | 21.9／−15.4／45.5 | 25.3／22.3／51.8 | 38.9／36.0／43.5 | 22.2／20.3／17.9 |

- **判定的字面套用**：「有看頭」要某元件在同一個（trace, 慢層, GPU）下、Cake、≥2 種 CPU 大小 ≥15％。rule=min 有 28 組成立（O-ORD 9、O-DROP 8、O-PF 11）；其中只用「oracle 與其最好對手的 ρ_w 都 <1」的格子也成立的：O-ORD 5、O-DROP 8、O-PF 7 組（O-DROP 和它在 NFS 的對手 2-hit 都寫得少，所以 NFS 上也不飽和）；完全不看 NFS、只看本地 SSD 與 b\*32：O-ORD 5、O-DROP 4、O-PF 7 組〔`x1_hot_groups.csv`；run 20261017-182005-p3x1-analyze〕。R1、R2 各自判也都是有看頭。
- 每個元件 96 格（rule=min）中 Cake 格 ≥15％ 的數目：O-ORD 30、O-DROP 25、O-PF 33、O-NOW 0；O-NOW 在 96 格的最大值 3.8％〔`x1_verdict.csv`〕。
- 最好的對稱對手（`x1_headroom_components.csv` best_rival）：O-ORD 的是 LRU2（128–256 GiB）或 GDSF（512–1024）；O-DROP、O-NOW 在 SSD／b\*32 是 LRU 本身，在 NFS 是 LRU+2HA 或 LRU+2HW；O-PF 的是 LRU 或 LRU+RPF（近期版預取只在 512 GiB 以上贏 LRU 一點點）。

### 4.3 值在哪裡：相對 LRU 的改善（只報告）

回來請求平均 TTFT 相對 LRU 的改善（％），R1、Cake、GPU 閒〔模擬，`x1_value_vs_lru.csv`，run 20261017-213401-p3x1-report〕：

| trace | 慢層 | CPU | LRU（s） | 最好的非 oracle | O-BEL | O-DROP | O-PF | O-ALL | O-ALL（s） |
|:--|:--|:--|--:|:--|--:|--:|--:|--:|--:|
| conversation | SSD | 128 | 0.644 | LRU2 11.8 | 33.6 | 25.9 | 48.8 | 47.7 | 0.337 |
| | | 256 | 0.589 | LRU2 12.3 | 33.5 | 32.2 | 43.6 | 38.0 | 0.365 |
| | | 1024 | 0.409 | GDSF 1.6 | 6.8 | 7.0 | 10.9 | 6.8 | 0.381 |
| | NFS | 256 | 1.890 | LRU+2HA 43.7 | 71.5 | 57.7 | 25.4 | 80.7 | 0.366 |
| toolagent | SSD | 128 | 0.403 | LRU2 5.8 | 16.8 | 17.2 | 25.4 | 22.2 | 0.313 |
| | | 1024 | 0.331 | GDSF 0.3 | −0.9 | −0.8 | 1.5 | −0.9 | 0.334 |

- 〔判讀〕三種資訊的對照：O-DROP 只用 1 個 bit（這個 session 之後會不會被讀），在 SSD 上拿到 Belady（確切時間）大約 3/4 到全部的改善（conversation 128 GiB：25.9 vs 33.6；256：32.2 vs 33.5）。也就是說，**降級這一側要的主要是「會不會回來」，不一定要「何時回來」**；預取那一側才需要時間。
- 〔判讀〕O-ALL 不等於各元件相加，有時還比單獨的 O-PF 慢（conversation SSD 128：47.7 vs 48.8）：三個貪婪元件疊起來不是整體最佳（B9 的「元件不能相加」）。
- 其他指標（同一格，conversation SSD 閒 128 GiB，LRU → O-ALL）：中位 0.445 → 0.178 s、p90 1.35 → 0.65 s、還原舊 context 部分 0.391 → 0.084 s、全部請求 TTFT 總和少 10.7％〔`x1_value_vs_lru.csv`〕。全部請求總和的改善遠小於回來請求的改善，因為第一輪的重算所有策略都一樣（D4 看到的稀釋）。

### 4.4 Cake 對只載入（只報告）

- 只載入 headroom >1% 的 142 格裡，Cake÷只載入 ≤0.5 的有 27 格（19％）；比值中位數 O-ORD 0.62、O-DROP 0.73、O-NOW 0.72、O-PF 0.77〔`x1_cake_vs_load.csv`，rule=min，run 20261017-182005-p3x1-analyze〕。
- 〔判讀〕Cake 讓空間小了大約 1/4 到 2/5，但多數格子沒有小到一半以下：依 §2.7 的報告規則，**「Cake 縮小預測的價值」不成立**。
- 只載入在 NFS 上會失控：讀取線飽和後排隊無上限（例如 conversation、R1、128 GiB、NFS、閒的 O-NOW 回來平均 1,339 s），這些格只能當「只載入在慢層不可用」的證據，不能當 headroom 的參考。

### 4.5 oracle 是不是上界（只報告）

- 175 列違反〔`x1_violations.csv`〕：Cake 格 69 列（O-NOW 44、O-DROP 11、O-ORD 5、O-PF 2、O-ALL 7），其餘 106 列在只載入的 NFS 格。
- O-NOW、O-DROP 的違反幾乎全在 NFS：對手 LRU+NONE（完全不寫）或 LRU+2HW 更快。〔判讀〕NFS 的讀取 0.331 GiB/s 大多比重算慢、寫入又讓讀取變慢 1.67 倍；oracle 只知道「誰會回來」、不知道「寫了值不值得」，所以寫給會回來的人反而虧。這是「只看資訊、不看成本」的 oracle 不是上界的實例。
- O-ORD／O-PF 只有 toolagent、R1、1 TiB、本地 SSD 的 7 列，最多慢 1.14％（O-BEL 0.334 s vs GDSF 0.330 s）。

### 4.6 FreeInference（事後加入，另列；不改原登記判定）

- 前 6 週、1,645 個 session、128,072 個回來請求（含工具觸發；使用者發起的另報）。四個元件 24 格（只有 Cake）的 headroom 最大值：O-ORD 0.22％、O-PF 0.16％、O-DROP 0.13％、O-NOW ≤0；只算使用者發起的回來：最大 1.06％〔模擬，run 20261017-213544-p3x1-analyze-fi、20261017-213548-p3x1-analyze-fiuser〕→ **FreeInference 版：死路**。
- 原因〔資料＋判讀〕：LRU 的回來平均 1.40 s（GPU 閒）幾乎全是新一輪的 prefill；ρ_w 最大 0.01、慢層幾乎沒被讀。這份 trace 只有 267 個帳號，同一時間活躍的 session 很少，128 GiB（約 11 個 89K-token 的 session）就裝得下。它是「規模小、context 長」的工作負載，不是容量壓力的工作負載。
- 含 FreeInference 的整體判定：有看頭（Mooncake 的組成立）〔run 20261017-213355-p3x1-analyze-all，`x1withfi_verdict.csv`〕。

### 4.7 事後穩健性：截尾（X9 提醒；不改原登記判定）

做法：O-DROP、O-NOW、O-ALL 的「之後再也沒人讀」只用在離 trace 最後一個請求還有 ≥H 秒的決定；最後 H 秒內一律當「會回來」（照 LRU 的機制保留、照寫）。對手不變。Mooncake 只有 3,537 s，所以 H＝3,600 s 等於完全不用這個 bit（O-DROP、O-NOW 變成 LRU；在對手就是 LRU 的格子 headroom 正好是 0.0，可當一致性檢查）〔模擬，run 20261017-213516-p3x1-hsweep → 20261017-221242-p3x1-posthoc，`x1_posthoc_checks.csv` check=dead_H1800／dead_H3600〕。

| 元件（每格取 R1／R2 較小，Cake） | 原登記 | H＝1,800 s | H＝3,600 s |
|:--|:--|:--|:--|
| O-DROP，conversation、本地 SSD、閒，128／256／512／1024 GiB | 22.8／28.2／18.6／7.0 | 9.3／12.1／9.4／3.6 | 0／0／0／0 |
| O-DROP，conversation、b\*32、閒 | 15.2／27.0／22.1／11.7 | 5.8／11.6／11.1／6.1 | 0／0／0／0 |
| O-DROP，conversation、NFS、閒 | −17.6／20.8／45.2／43.2 | −49.4／−35.5／13.3／22.1 | −71.2／−77.7／−19.6／−0.9 |
| O-NOW，所有本地 SSD／b\*32 格 | ≤3.9 | ≤1.3 | 0 |
| O-ORD、O-PF | （不用這個 bit，和原登記相同） | 同左 | 同左 |
| **判定** | 有看頭（O-ORD、O-DROP、O-PF） | **有看頭**（21 組；O-DROP 只剩 1 組：conversation、NFS、GPU 忙，512／1024 GiB） | **有看頭**（20 組；只有 O-ORD、O-PF） |

- 〔判讀〕O-DROP 的空間有一大半來自「trace 最後一段做的丟掉決定，代價（之後回來要整段重算）不在 trace 裡」。O-ORD（Belady）只降級、不丟：trace 結束後才回來的 session 在真實的未來裡也一樣是「最晚回來」，所以 Belady 的順序不受截尾影響；殘留的偏差是「它們之後從慢層還原的代價沒算進來」，這對所有策略都一樣不算，但 Belady 會優先降級它們〔判讀，NOT_MEASURED〕。O-PF 騰空間時也會先降級這些 session，同一個殘留偏差。

### 4.8 事後加入的對手與容量軸（X2 提醒；只報告）

- **FREQ2Q**（session 版 2Q：先比到目前為止的請求數、再比閒置時間）：在 Cake、GPU 閒的 24 格（×2 規則）都比原登記最好的順序對手慢，R1 conversation 本地 SSD 128→1024 GiB：−2.6／−5.0／−8.0／−12.0％；NFS 最多 −45％。96 格中 0 格成為最好對手；O-ORD 的判定與每格 headroom 都不變〔run 20261017-215113-p3x1-postsweep、20261017-221242-p3x1-posthoc〕。〔判讀〕Mooncake 上「請求數多的先留」會一直留著早期很活躍、後來不再回來的 session（沒有老化）；原登記的 LRU-2 已經是 session 層級的頻率感知對手，而且小容量時是最好的。X2 在 Bailian 上看到的頻率訊號，在這裡沒有變成比 LRU-2 更好的規則。
- **容量軸「CPU 裝得下幾分鐘的歷史」**（LRU 下，session 從回合結束到被降級的平均時間）：

| trace | 128 GiB | 256 | 512 | 1024 |
|:--|--:|--:|--:|--:|
| conversation | 0.46 分 | 0.93 | 1.96 | 4.30 |
| toolagent | 0.49 | 0.99 | 2.12 | 4.71 |

來源：`x1_sweep_post.csv` 的 cpu_residence_s_mean（本地 SSD、GPU 閒、Cake；LRU 的狀態演變和時間設定無關，兩種重建規則相同）〔模擬〕。對照回來間隔中位 129–147 s（§4.1）：128–512 GiB 時，LRU 留住的歷史比一般的回來間隔短，這正是預測有空間的區域。

### 4.9 兩份 Mooncake trace 的重疊（X5 提醒）

- conversation 12,031 個請求中，10,938 個（90.9％）在 toolagent 有相同的（timestamp、input_length、output_length），而且 hash_ids 的個數也相同〔資料，run 20261017-221242-p3x1-posthoc，`x1_posthoc_checks.csv` check=trace_overlap〕。
- 所以 conversation 與 toolagent 不是兩份獨立證據。判定改用「只算 conversation」也成立（原登記版與兩個截尾版都有 conversation 的組過門檻：例如 O-PF、O-ORD 在 conversation 本地 SSD 閒的 128／256／512 GiB 都 ≥15％）。

## 5. 前作

這一輪我沒有親自打開任何論文原文；下面的頁碼來自文獻 agent 的 `../research/LIT_AND_TRACES.md`（它讀了原文）與 D4 §5，標〔引自 LIT／D4，未親自查證〕。演算法本身（Belady、LRU-K、ARC、S3-FIFO、GDSF、Aggressive 預取、Belady-Size）是照我記得的定義實作，標〔未查原文〕，引用前要核對。

| 前作 | 和本方向的關係 | 出處標記 |
|:--|:--|:--|
| D4（本 repo） | block 層級、總 TTFT：oracle 對最好延後版只差 0.015％／0.002％。本方向改成 session 層級、只算回來請求後，差距變成 10–60％——**差別主要來自指標與決策單位**，不是新的資訊 | 〔本 repo，D4 §0〕 |
| Ask-the-Tool（arXiv 2609.18849） | 加上 DRAM 層後，oracle 的 p90 改善從 27.7％降到 22.6％；兩個已發表的預測器只有 HBM 時比 LRU 差 22–23％ | 〔引自 LIT p.13，未親自查證〕 |
| CacheScout（arXiv 2608.14624） | 單獨預取最多 +1 個百分點，增益主要來自逐出；和本方向「預取最大」的結果相反〔判讀：他們是 GPU↔CPU 的重算暖機，沒有慢層頻寬的取捨〕 | 〔引自 LIT p.9–10，未親自查證〕 |
| WhereKVLive | 在爭用的連結上，連 oracle 預取都沒有一格贏過不預取；本方向在 NFS（讀取線飽和）上 O-PF 的值確實最小（128 GiB 只有 9％），和它一致 | 〔引自 LIT 的卡片，未親自查證〕 |
| LPC（NeurIPS'25） | 只預測「會不會續聊」的分類器，補回 LRU→oracle 差距的 22–66％；本方向 O-DROP（1 bit）在 SSD 上拿到 Belady 改善的 3/4 以上，和「1 bit 就有大部分價值」一致 | 〔引自 D4 §5，LPC 原文 p.23〕 |
| Pensieve（EuroSys'25） | 本方向的 PENS 對手（session 版）；在小容量時輸給 LRU-2 | 〔引自 D4 §5〕 |
| Belady-Size（LRB, NSDI'20 用的離線上限） | O-ORD 族的第二個 oracle；多數格裡 Belady 本身就比較好 | 〔未查原文〕 |
| Aggressive 預取（Cao、Felten、Karlin、Li，SIGMETRICS'95） | O-PF 的規則 | 〔未查原文〕 |
| LRU-K（O'Neil 等 SIGMOD'93）、ARC（FAST'03）、S3-FIFO（SOSP'23）、GDSF（Cherkasova 1998） | 對手，session 版；S3-FIFO 的參數照 `m8_trace_oracle.run_s3fifo` | 〔未查原文〕 |

## 6. 偏離與問題

**失敗（照實列，完整錯誤）**

| run_id | 指令 | 錯誤 | 處理 |
|:--|:--|:--|:--|
| 20261017-213346-p3x1-analyze-fi、20261017-213350-p3x1-analyze-fiuser | `x1_sim.py analyze --sweep x1_sweep_fi.csv --traces freeinference --out-prefix x1fi`（另一個加 `--metric mean_ret_user --out-prefix x1fiuser`） | `File ".../code/p3x/x1_sim.py", line 1223, in cmd_analyze: vr = RR[RR.load_gt1pct] ... AttributeError: 'DataFrame' object has no attribute 'load_gt1pct'`（exit 1） | FreeInference 只跑 Cake，沒有只載入格，比值表是空的。錯在判定 CSV 寫完之後的列印步驟。加防呆重跑 |
| 20261017-213519-p3x1-analyze-fi、20261017-213524-p3x1-analyze-fiuser | 同上 | 下一行同樣的問題：`print(... int((vr.ratio <= 0.5).sum()))` → `AttributeError: 'DataFrame' object has no attribute 'ratio'`（exit 1） | 再加防呆，重跑 20261017-213544、20261017-213548（exit 0）。四個失敗的 run 寫出的 `x1fi_*`／`x1fiuser_*` 已被成功的那次覆蓋；文件只用成功那次 |
| （環境 agent）20261017-173416-p3-0-data | 下載 Mooncake | stdout 記 `[mooncake] FAILED(rc=1)`，但三個 trace 與 arxiv trace 的 git-blob 都比對成功；目錄裡沒有 `Mooncake-FAST25.pdf`〔判讀：失敗的是 PDF〕 | 用 trace 檔；sha256 另記（conversation `b8cbb061…`、toolagent `48a2db1a…`），列數與 D4 相同 |

另外：主 session 說我在 18:2x 因 API 用量上限中斷。中斷前啟動的 20261017-181951-p3x1-fisweep 與 20261017-182005-p3x1-analyze 都已跑完（exit 0、stderr 空），沒有重跑。

**偏離登記、事後加入（不改原登記判定）**

- §2.9 的 R2 重建規則與 FreeInference（加入時已看過的結果照實寫在 §2.9）。
- 截尾穩健性（§4.7，X9 提醒）、session 版 2Q 對手 FREQ2Q（§4.8，X2 提醒）、「CPU 裝得下幾分鐘的歷史」容量軸（§4.8）：都是看過主結果之後才加的。
- 自檢只對 R1 跑了 PRED≡LRU 與多設定一致性；R2、FreeInference 沒有跑（同一份程式，只換重建結果）。
- 主 sweep 之後改過 `x1_sim.py`（FreeInference、`-H` 變體、FREQ2Q、停留時間欄）。檢查：改完後重跑的 LRU（20261017-215113-p3x1-postsweep）和主 sweep 的 LRU 在 192 格的 mean_ret 逐格相同〔run 20261017-221242-p3x1-posthoc〕。
- `x1fi_hot_groups.csv`、`x1fiuser_hot_groups.csv` 原本是空檔（沒有任何組過門檻），我手寫了一列說明（run_id 指回那次 analyze），讓檔案有 run_id、ts 欄。
- FreeInference 的 58 列「oracle 比對手慢」（`x1fi_violations.csv`）最多慢 0.013％〔`x1fi_violations.csv` 的 oracle_slower_by 欄〕，不影響判定。
- 截尾穩健版的 sweep 用的是加了 `-H` 變體之後的程式；它的對手列沿用主 sweep 的 `x1_sweep.csv`（對手的程式路徑沒有改，但 code_sha 不同；兩次的程式快照都在各自 run 目錄的 `code_snapshot/`）。

**模型的限制（會影響數字的方向）**

1. **一次一個請求、GPU 不排隊**（和 D4、第一階段相同）。Mooncake 實際 3.4／6.7 req/s，一張 GPU 跑不完；併發與排隊是 X4 的範圍。
2. **`free` 語意**：降級時 CPU 空間立刻釋放、位置立刻算在慢層。NFS 上 ρ_w>1（寫不完），還沒寫完的資料就能讀，物理上不成立；所以 NFS 格只能當上限參考（判定的組在 SSD／b\*32 也成立，見 §4.2）。
3. **只載入在 NFS 上排隊無上限**（§4.4），那些格的 headroom 和比值沒有意義。
4. **b\*32 的寫入頻寬＝讀取、k_read＝1** 是假設（限速層沒量過寫入與干擾）。
5. **慢層容量無上限**；O-DROP、O-NOW 的價值因此只來自 CPU 空間與寫入干擾。
6. **o200k_base 的 512 token 當成 Llama 的 512 token**（FreeInference），比值 NOT_MEASURED。
7. **f(i) 在 i ≥ 80 線性外插**；Mooncake 最長 246 個完整 block（`x1_trace_char.csv`）；FreeInference 第 1 週 prompt 中位 89,242 token（文獻 agent，`../research/cards/P3_FreeInference_trace.md`）≈ 174 個 chunk〔算術〕，大部分落在外插區〔外插〕。
8. **前綴命中的假設（X6 提醒）**：模擬器裡有存的 block 一定是一段前綴（refcount 保證，每步斷言），所以「頭不在、尾在」的情況不會出現；降級／丟掉都是以 session 為單位整段移動「只有它持有的後綴」。真實系統（vLLM、LMCache、SGLang、TRT-LLM、Dynamo）是**逐 block 從尾端淘汰、只認最長前綴**。本模擬器沒有「session 內只丟尾端幾個 block」的動作；要做「真實系統式淘汰」的敏感度，需要改成 block 層級的淘汰，不便宜，留到 §7。〔判讀〕Cake 要的是「前段重算、後段載入」，真實系統是「先丟尾端」，方向相反；block 層級淘汰下，被丟掉的尾端正好是 Cake 最想載入的部分，headroom 可能更大或更小，NOT_MEASURED。
9. **INT4 反量化只放在載入線上**（m10_iowall 的模型）；降級時的量化成本（GPU 上）沒有算進任何人的 TTFT。
10. **oracle 不是上界**（§4.5）：O-NOW／O-DROP 只知道誰會回來，不知道寫不寫划算。

## 7. 如果要繼續，下一步

1. **先確認不是模型造成的**（最便宜、最該先做）：
   - block 層級的「真實系統式淘汰」敏感度（§6 第 8 條）：session 內從尾端逐 block 淘汰、只認最長前綴，看 O-PF／O-DROP／O-ORD 的 headroom 還在不在。
   - 把 X4 的併發（GPU 排隊、SSD 寫入佇列有上限）接上同一套策略；特別是 O-PF，它在 NFS 上的慢層讀取利用率 ρ_r 已經到 0.49–0.99（LRU 只有 0.16–0.71；`x1_sweep.csv` 的 rho_r 欄）。
   - 3090 常數（X7 出來後）換掉 MI300X 常數重跑；判定的組是否還在。
2. **猜得到嗎**（和 X2、X3 交集）：
   - 降級這一側主要要「會不會回來」1 個 bit（§4.3）。把 O-DROP 換成「部分 oracle」：只知道 H 秒內回不回來（H＝60、300、1800 s），以及加雜訊的 oracle，畫出「預測準確度 → headroom」曲線；X2 的 C-index 能換算到這條曲線上的哪裡。
   - 預取這一側需要時間：用 X3 的現成提示（打字、工具呼叫）做「只在回來前 x 秒知道」的 O-PF，看拿得到多少。
3. **FreeInference 為什麼是 0**：用同一份 trace 把多週疊在一起（同一個時間軸上同時重播多週、或縮小 CPU 到 16–64 GiB），看是「沒壓力」還是「長 context 讓 Cake 本來就夠快」。
4. **不值得做**：O-NOW（不回來的不寫慢層）——所有格子 <5％，NFS 上比「完全不寫」還慢。
