# X9　其他想法：延後設計下，還有哪些問題需要額外資訊？

> 方向登記：[`README.md`](README.md) §2 X9。共同規則：同檔 §1。
> 標記：〔實測〕＝本次從 trace／資料算出來、附 run_id；〔算術〕；〔MI300X 常數〕；〔原文 p.X〕＝讀了原文；〔摘要〕＝只看摘要或文件頁；〔文件〕＝官方文件；〔判讀〕；〔假設〕。
> 本文件是 self-review，不是零前文審查。

---

## 0. 一句話結論＋判定

**列了 20 個想法：0 個有看頭、10 個可能、10 個死路（§4.1；I20 是看到 FreeInference 結果後才加的）。前 3 名的測試照字面都判「可能」，沒有一個變成第三階段的新研究問題。但三個都找到 X1／X5／X8 模擬器的前提問題，影響 P3-3 的判定：**

| 測試 | 判定（照 §2.2 字面） | 關鍵數字 | 對第三階段的意思〔判讀〕 |
|:--|:--|:--|:--|
| T1 回合結束就死掉的 KV | **可能** | 思考模型 trace：父 KV 的 ΣD⁻/ΣS＝**5.30%**（下界；上界 9.90%）；換成「父的輸出」當分母是 28.2–52.6%。traceA、coder ≈0；四個 trace **一個兄弟都沒有**（分岔率 0）〔實測 `20261017-174829-p3x9-census`、`20261017-175022-p3x9-extra`〕 | 只有思考模型有量，而且只有 509 個回來的 session（<1,000）。「死掉」這件事是永久的，延後版只要有人告訴它就看得到，所以這是要不要開提示 API 的問題，不是預測問題。前作 Lachesis 已經做了 |
| T2 跨 session 共用前綴 | **可能** | 回來請求的前綴中，跨 session 共用的位元組占比：**14.8%／17.0%／20.6%**（traceA／thinking／coder）。用 session 當單位算工作集，會比不重複 block 多算 **1.27–2.60 倍**〔實測 `20261017-174829-p3x9-census`、`20261017-175022-p3x9-extra`〕 | 共用的部分在降級當下就看得到，不需要預測。X1 已經用 refcount；用 session 當單位的模擬（例如 X8）若用「CPU 占工作集 %」當容量，工作集會差 1.3–2.6 倍 |
| T3 1–2 小時視窗把「晚點回來」當成「不回來」 | **可能** | WildChat-1M（837,989 個對話）：1 小時視窗標成「不回來」的回合結束裡，其實有回來的占 **M_1h＝14.6%**（2 小時 9.8%）。只算至少 2 輪的對話：30.2%／21.5%〔實測 `20261017-174829-p3x9-window`〕 | X1 的 O-丟掉／O-不寫慢層、X5 與 X8 的「不回來的直接丟」、D4 的「75–78% 寫了沒人讀」都用 1–2 小時的 trace。這些 oracle 會因此偏樂觀，P3-3 要加設限保護 |

**FreeInference 版（登記之後才加，§2.4；照 README §4，總結以上面的原登記版為準，這一版是「下一輪要確認的線索」）**〔實測 run `20261017-181049-p3x9-freeinf`〕：

| 測試 | 判定（照 §2.4 字面） | 關鍵數字 |
|:--|:--|:--|
| T1-FI | **死路** | ΣD⁻/ΣS＝1.49%（上界 1.54%）。但只報告的「死掉 ÷ 新寫入」＝**57.0%**（上界 58.9%）：agent session 每一步都重送整段 context，ΣS 是 Σ新寫入的 38 倍（〔算術〕203.9B ÷ 5.35B token），原登記的分母在 agent trace 上會把死掉的比例稀釋掉（§6 第 11 點）。死掉的質量 37% 來自中途改 system prompt、19% 來自改工具定義、壓縮只占 2.4% |
| T2-FI | **可能** | 位元組加權共用占比 25.7%；**跨使用者只有 1.7%**；session ÷ 不重複 block＝1.17 |
| T3-FI | **有看頭** | **M_1h＝47.5%**（2 h 35.6%）；只算「下一步由使用者觸發」的回來：36.8%（2 h 27.3%）；session 不跨週，所以是下界 |

- **最值得追的一個〔判讀〕**：**T3 的視窗截斷**。FreeInference 版更嚴重（1 小時視窗標「不回來」的請求結束裡，近一半其實會回來）。它不是新的研究問題，但它直接決定 P3-3 的 oracle 是不是上界，以及 O-丟掉／O-不寫的 headroom 是不是被灌水。下一步也最便宜：在 X1 的模擬裡，只評估離 trace 結束還有至少 H 秒的決定（H＝1,800 s 或 3,600 s），或把「視窗內沒回來」當設限，再比一次 headroom（§7）。
- 第二值得追的是 T2 附帶的「session 與 block 的工作集差 1.3–2.6 倍」：它不改任何判定，但各方向的 CPU 容量若用「占工作集 %」，要說明用哪一種單位（X1 用 refcount，已經是 block 層級）。
- 第三個給 X1 的提醒（T1 附帶）：我用較寬的重建規則在 Mooncake 找到的父子邊裡，conversation 有 10.7%、toolagent 有 21.7% 是「下一輪輸入的增加量 < 上一輪輸出」〔實測 `t1_census.csv`，只報告〕；X1 §2.1 的規則 `input(r) ≥ input(p)+output(p)` 不會把這些接成同一個 session。這些邊可能是真的剝除，也可能是我接錯父，分辨不出來。

---

## 1. 問題與要證明什麼

X1–X8 問的是「猜對誰回來、何時回來值多少、猜不猜得到」。X9 找**其他**「延後設計（BF16 進 CPU、滿了才降級、降級時 INT4、回來時 Cake 還原）下需要額外資訊」的問題，每個給快速判定；挑前 3 名各做一個最便宜的測試（只用 CPU、固定規則或無參數統計，不訓練）。

選前 3 名的理由（看結果之前，在 §2 寫判準時就定了）：
1. 現在就有 ≥1,000 個 session 的真實資料可以測（Bailian 有明確的 `parent_chat_id`；WildChat 有一年多的時間戳）。
2. 只用 CPU、幾分鐘跑完。
3. 不重複 X1–X8 的測試，而且**可能推翻它們模擬器的前提**（下面是寫完 §4 後再對照各文件 §2 的字句，2026-10-17T10:01Z 查）：
   - 回來要還原的是「上一輪輸入＋輸出」：X2 §2.1、X3 §2.1 這樣定義；X1 §2.1 的 session 重建要求 `input(r) ≥ input(p)+output(p)`，下一輪輸入變短的就接不上（T1 檢查）；
   - 用 session 當降級單位：X8 §2.1；X1 §2.1 已經改用 refcount 處理共用前綴（T2 檢查）；
   - 「trace 結束前沒回來＝永遠不回來」：X8 §2.1、X5 §2 的 oracle「不回來的直接丟」（T3 檢查）。
   第三階段的口號是「先找出問題」，前提錯了，headroom 就會跟著錯。

其他 17 個想法只查前作、找資料、給判定（§4.1）。

---

## 2. 判準（開跑前寫）

**寫於**：2026-10-17T09:44:06Z（`date -u`）開始寫，寫完時間見本節最後一行。寫的時候只看過：Bailian、Mooncake 每檔前 2–3 列的結構與環境 agent 的單位檢查（`datasets/*/README.txt`）、WildChat 的 parquet schema、CacheWise 一個 raw 檔的前 3,000 字元（只看欄位名）。**沒有算過任何下面要判的數字。**

### 2.1 每個想法的判定（不測試的那些）

- **有看頭**：(i) 在延後設計＋Cake 下，這個問題的答案需要一個「寫入／降級當下拿不到、或要預測」的資訊；(ii) 查到的前作沒有在多層＋讀取端自適應的設定下做過；(iii) 有可取得的真實資料能測。三條都成立。
- **死路**：(i) 不成立（延後版在決定當下就看得到同樣的資訊，或 Cake 讓它不重要），或有前作已經做掉而且我們沒有多一個維度，或資料／硬體上做不到。
- 其他＝**可能**。
- 這是**判讀**，不是量測；每一列都寫一句理由與來源。

### 2.2 前 3 名的測試（每個先定義、再寫判準；只用 CPU）

共同：
- 資料只讀 `/ssd8/hungwei/paper-hkv/datasets/`，用 `python -I`。
- 規則 6 斷言寫在載入函式裡（見每個測試）。
- 「獨立 session」＝Bailian 的根鏈（沿 `parent_chat_id` 走到 `-1`）、WildChat 的 `conversation_hash`。README §1 的門檻：至少 1,000 個獨立 session 才能讓判定變「有看頭」。

#### T1　回合結束就已經死掉的 KV（上下文變動普查）

- **問題**：「寫入簡單」會把每一輪的輸入＋輸出 KV 全部寫進 CPU。如果下一輪的輸入**不是**上一輪輸入＋輸出的延伸（思考內容被剝掉、前端截斷、編輯舊訊息、按重新生成），一部分剛寫進去的 KV 在回合結束時就已經不會再被當成前綴用到。LRU 會把這些最新寫入、卻已經死掉的 KV 留在 CPU 最久。這一部分有多大？是不是在回合結束時就能用固定規則認出來（不需要預測）？
- **資料**：Qwen-Bailian traceA（To-C）、traceB（To-B API）、thinking、coder（`parent_chat_id` 明確給出父子）。
- **規則 6 斷言**：(a) `len(hash_ids)==ceil(input_length/16)` 的比例 ≥0.99；(b) 父子邊中 `timestamp_子 ≥ timestamp_父` 的比例 ≥0.99；(c) `turn_子 == turn_父+1` 的比例照實報告（不設門檻，但 <0.95 要在 §6 寫）。
- **定義**（每條父→子邊，p＝父、c＝子，block＝16 token）：
  - L＝`hash_ids` 的最長共同前綴（block 數）；n_p＝`len(hash_ids_p)`。
  - 「父輸入整份延續」＝L ≥ n_p − 1（上游 FAQ Q2：最後一個 block 可能因 padding 改變）。
  - Δ＝`input_length_c − input_length_p`（token）。
  - **APPEND**：整份延續且 Δ ≥ `output_length_p`。
  - **STRIP**：整份延續且 Δ < `output_length_p`（至少有 `output_length_p − max(Δ,0)` 個父的輸出 token 沒有被帶到下一輪）。
  - **REWRITE**：不是整份延續（子在父的輸入中間就分岔：編輯、截斷、檢索內容換掉等）。
  - 父在回合結束時寫入的 KV：S_p＝`input_length_p + output_length_p` token。
  - 每個子能重用的父 KV（token）：樂觀值 R⁺＝APPEND：S_p；STRIP：`input_length_p + min(output_length_p, max(Δ,0))`；REWRITE：min(16·L, `input_length_p`)。悲觀值 R⁻＝APPEND：S_p；STRIP、REWRITE：min(16·L, `input_length_p`)（輸出只要被剝掉任何 token，後面的位置就全部錯開，輸出 KV 不能當前綴）。
  - 每個「至少有一個子」的父：死掉的下界 D⁻_p＝S_p − max_c R⁺；上界 D⁺_p＝S_p − max_c R⁻。
  - **主指標**：每個 trace 的 ΣD⁻ ÷ ΣS（只算至少有一個子的父；沒有子的父是「不回來或設限」，不是「死於變動」，另外報告數量）。
  - 另報：ΣD⁺ ÷ ΣS；各類別的邊數比例；**分岔率**＝至少有一個兄弟（同父）的子請求比例；其中輸入完全相同（`hash_ids` 相同）的兄弟比例（＝重新生成）；依 `type` 分開的 ΣD⁻/ΣS。
- **判準**：
  - **有看頭**：至少一個「有子的父 ≥1,000 個、且獨立 session ≥1,000」的 trace，ΣD⁻/ΣS ≥ 20%。
  - **死路**：四個 trace 的 ΣD⁻/ΣS 都 <5%，**而且**分岔率都 <5%。
  - 其他＝**可能**。
  - 附帶判讀（不改判定）：D⁻ 的質量裡，STRIP 類占多少（STRIP 若來自 chat template 剝思考內容，是回合結束時就確定的事，不用預測）。

#### T2　回來的 session 有多少 KV 其實是別人的（跨 session 共用前綴）

- **問題**：X1／X5／X8 用 session 當降級單位。如果回來請求要還原的前綴有很大一部分是很多 session 共用的（system prompt、熱門文件），(a) session 層級的降級會重複計算容量、把熱門 block 跟著冷 session 一起搬下去；(b) 共用的前段很可能還在 GPU／CPU 裡，Cake 的重算線從共用前綴之後才開始，起點的 f(i) 更貴。這一部分有多大？
- **資料**：同 T1 的四個 Bailian trace（主）；Mooncake conversation、toolagent（只報告：沒有明確父子，用「前一個 `hash_ids` 是前綴的請求」當父，block＝512 token；斷言 `len(hash_ids)==ceil(input_length/512)` ≥0.95）。
- **定義**：
  - 每個 block id 的「session 數」＝整個 trace 中含這個 id 的不同根 session 數（用了未來資訊；這是描述統計，不是策略）。
  - 回來請求 c（有父的請求）：L_c 同 T1；共用前綴 P_c＝從 block 0 起連續、session 數 ≥2 的 block 數。共用占比＝min(P_c, L_c) ÷ L_c（L_c＝0 的不算）。
  - **主指標**：每個 trace 的位元組加權共用占比 Σmin(P_c,L_c) ÷ ΣL_c；同時報中位數與 p90。
  - 另報〔算術，MI300X 常數〕：用 f(i)（`calib_c1.csv`，block ≥80 線性外插）與 b\*=32 層（1.47 GiB/s）、NFS（0.331 GiB/s），把每個回來請求的前綴換成 512-token chunk，算「共用前綴已在 GPU（免費）」與「整段從慢層還原」的 Cake 時間差，報中位數。
- **判準**：
  - **有看頭**：至少一個獨立 session ≥1,000 的 Bailian trace，位元組加權共用占比 ≥30%。
  - **死路**：四個 Bailian trace 都 <5%。
  - 其他＝**可能**。Mooncake 只報告。

#### T3　1–2 小時的 trace 把「晚點回來」當成「不回來」（視窗截斷）

- **問題**：X1 與 D4 用 Mooncake（約 1 小時）與 Bailian（2 小時）。在這麼短的 trace 裡，視窗結束前沒回來的 session 會被 oracle 當成「永遠不回來 → 直接丟／不寫慢層」。真實世界裡其中有多少其實是晚一點才回來？這是「oracle 是不是上界」與「O-丟掉值多少」的前提。
- **資料**：WildChat-1M（14 個 parquet，一年多的真實時間戳）。只讀 `conversation_hash`、`timestamp`、`turn`、`conversation` 裡的 `role` 與 `timestamp`。
- **規則 6 斷言**：(a) 頂層 `turn` ＝ user 訊息數的比例 ≥0.99；(b) 有時間戳的訊息在同一對話內不遞減的比例 ≥0.99；(c) 頂層 `timestamp` ＝最後一個有時間戳訊息的時間戳的比例照實報告。哪些 role 有時間戳照實報告。
- **定義**：
  - 一個對話的「回合結束時刻」e_j＝第 j 個 assistant 訊息的時間戳；「回來時刻」＝第 j+1 個 assistant 訊息的時間戳（若 user 訊息有時間戳，改用第 j+1 個 user 訊息，§6 寫明用了哪個）；間隔 g_j＝兩者之差；最後一輪 g＝∞（資料裡沒有回來）。
  - 視窗 W ∈ {1 h, 2 h}：把每個回合結束時刻視為落在一個長 W 的 trace 視窗裡的均勻位置，剩下的視窗長度 U·W，U～Uniform(0,1)。被視窗標成「不回來」的機率＝min(1, g/W)（g＝∞ 時為 1）。
  - **主指標**：M_W＝Σ_{g 有限} min(1, g/W) ÷ Σ_{全部} min(1, g/W)＝「被視窗標成不回來的回合結束」裡，其實有回來的比例（期望值，無參數）。
  - 另報：有回來的間隔中 >1 h、>2 h、>24 h 的比例；M_W 只算 ≥2 輪的對話的版本。
- **判準**：
  - **有看頭**（＝這是第三階段必須處理的真問題）：M_1h ≥ 20%。
  - **死路**：M_1h < 5%。
  - 其他＝**可能**。
  - 已知偏誤（寫在前面）：WildChat 的介面若不能在重新整理後接續舊對話，長間隔會被低估，M_W 是下界〔判讀〕；它是免費代理的使用者，不是付費產品。

### 2.3 執行

- 程式：`code/p3x/x9_census.py`（T1、T2）、`code/p3x/x9_window.py`（T3）、`code/p3x/x9_arith.py`（T2 的 Cake 算術）。
- 結果：`results/p3_explore/x9/`（每個 CSV 有 `run_id`、`ts`）；原始 log：`/ssd8/hungwei/paper-hkv/runs/<RUN_ID>/`。
- 跑之前不看任何中間數字來改上面的定義或門檻；程式錯誤若在看到結果之後才修，照實寫在 §6。

（§2 寫完：2026-10-17T09:45:06Z，`date -u`。之後沒有改。）

### 2.4 FreeInference 版（登記之後才加；2026-10-17T10:02:37Z 開始寫，`date -u`）

- **為什麼加**：主 session 在 README §4（10:01Z）補了 FreeInference agentic trace（12,002 個 session、16 週、16-token 鏈式 block、每個請求有作者標的 `transition`／`cause`）。照 README §4 第 1 點：判定列兩個版本，**總結以原登記版（§2.2）為準**，FreeInference 版當「下一輪要確認的線索」。
- **寫的時候已經知道的**：§2.2 三個測試在 Bailian／WildChat 上的結果（§4.2–4.4）；FreeInference 的欄位名、`transition` 的值域（append／switch／mutation）、`cause` 的值域（compaction、other、system_prompt、injection、dropped_turns、tool_definitions、tool_result）、第 1 週的請求數（10,738）。**沒有算過 FreeInference 的任何下面要判的數字。** 門檻照抄 §2.2，不改。
- **資料**：`/ssd8/hungwei/paper-hkv/datasets/freeinference_agentic_trace/release/*/processed/{requests,sessions}.parquet`（16 週）。session＝`session_id`；同一 session 的請求依 `step` 排序。
- **規則 6 斷言**：(a) 單位：`role_tokens` 五欄加總 ≥160 的請求中，16·`len(block_ids)` ÷ `role_tokens` 加總落在 [0.9, 1.1] 的比例 ≥0.95（README：兩者都是 o200k_base，block＝16 token、最後不滿的 block 丟掉）；(b) 標籤：作者標 `transition=='append'` 的相鄰請求中，`block_ids` 共同前綴 ≥ 前一個的長度 − 1 的比例 ≥0.95；(c) 同一 session 內 `step` 嚴格遞增、`start_ms` 不遞減的比例 ≥0.99。任何一條不過就停、照實記錄。
- **T1-FI**：父子邊＝同一 session 的相鄰請求（p＝第 k 步，c＝第 k+1 步）。分類同 §2.2，但輸入長度一律用 16·`len(block_ids)`（o200k）：in_p＝16·n_p、Δ＝16·(n_c − n_p)；out_p＝`completion_tokens_p`（**供應商的 tokenizer，和 o200k 不同；這是近似**）。S_p、R⁺、R⁻、D⁻、D⁺ 定義同 §2.2。主指標 ΣD⁻/ΣS。判準：**有看頭** ≥20%（至少 1,000 個有邊的 session）；**死路** <5%；其他＝可能（分岔率在這份資料不適用：session 是線性串起來的）。只報告：D⁻ 依作者的 `transition`／`cause` 分開；我的分類和作者標籤的一致程度。
- **T2-FI**：每週分開算（README：block id 只在同一週內有意義）。共用 block＝同一週內出現在 ≥2 個不同 `session_id` 的 block。回來請求＝T1-FI 的子；L、P、共用占比同 §2.2。主指標：位元組加權 Σmin(P,L) ÷ ΣL（16 週合計）。判準：**有看頭** ≥30%；**死路** <5%；其他＝可能。只報告：改用「≥2 個不同 `user_id`」的跨使用者版本；session 層級工作集 ÷ 不重複 block（每週）。
- **T3-FI**：回合結束＝每個請求的 `end_ms`；回來＝同一 session 下一個請求的 `start_ms`；session 的最後一個請求 g＝∞。M_W 公式同 §2.2，W＝1 h、2 h。判準：**有看頭** M_1h ≥20%；**死路** <5%；其他＝可能。只報告：只算「下一個請求由使用者觸發（`step_trigger=='user'`）或是最後一個」的回合結束（＝人回來，不含工具回來）。已知偏誤：session 不跨週（跨週的會被切成新 session），所以 M_W 是下界。
- 程式：`code/p3x/x9_freeinf.py`。

（§2.4 寫完：2026-10-17T10:03:00Z，`date -u`。之後沒有改。）

---

## 3. 做了什麼（指令、run_id、產出）

全部用 `TIARA_NO_GPU=1 bash code/run3090.sh <短名> <指令>`（CLAUDE.md §4.1 的記錄殼；`CUDA_VISIBLE_DEVICES=""`、`HIP_VISIBLE_DEVICES=""`）。Python＝`/ssd8/hungwei/paper-hkv/venv/tiara/bin/python -I`。run 目錄在 `/ssd8/hungwei/paper-hkv/runs/<run_id>/`（`cmd.sh`、`context.txt`、`stdout.log`、`stderr.log`、`exit_code`）。

| run_id | 指令（`cmd.sh`） | 用途 | 產出 | exit |
|:--|:--|:--|:--|:--|
| `20261017-174657-p3x9-census` | `python -I code/p3x/x9_census.py --out results/p3_explore/x9 --rundir <run>` | T1、T2 第一次 | **失敗**：traceB 觸發規則 6 (b) 的斷言（§6 第 1 點）；只留下 traceA 的逐邊檔 | 1 |
| `20261017-174829-p3x9-census` | 同上（修好 0／0 的錯誤） | T1、T2 | `results/p3_explore/x9/t1_census.csv`、`t1_by_type.csv`、`t2_shared_prefix.csv`；逐邊檔 `t1_edges_*.csv.gz`、`t2_returns_*.csv.gz` 在 run 目錄 | 0 |
| `20261017-174829-p3x9-window` | `python -I code/p3x/x9_window.py --out results/p3_explore/x9 --rundir <run> --procs 4` | T3 | `t3_window_censoring.csv`、`t3_gap_distribution.csv`；`t3_gaps_s.npy` 在 run 目錄 | 0 |
| `20261017-175022-p3x9-extra` | `python -I code/p3x/x9_extra.py --census-run <174829 census> --out results/p3_explore/x9` | 只報告：T1 改用父輸出當分母；session vs 不重複 block 的工作集 | `t1_breakdown.csv`、`t2_session_vs_block_workingset.csv` | 0 |
| `20261017-175327-p3x9-gapcheck` | `python -I code/p3x/x9_gapcheck.py --wildchat-gaps <window run>/t3_gaps_s.npy --out results/p3_explore/x9` | 只報告（事後）：WildChat 與 Bailian 在 1 小時內的間隔分布像不像 | `t3_gap_consistency.csv` | 0 |
| `20261017-175449-p3x9-forkcheck` | `python -I code/p3x/x9_forkcheck.py --out results/p3_explore/x9` | 只報告（事後）：分岔是不是被記成新的根 session | `t1t2_forkcheck.csv` | 0 |
| `20261017-175709-p3x9-xcheck` | `python code/p3x/x9_crosscheck.py --out results/p3_explore/x9` | 檢查 T2 算術用的 Cake 兩條線函式和 `m7_sim.sim_restore` 一致 | `t2_cake_crosscheck.csv`：最大相對差 3.9×10⁻¹⁶（16 格） | 0 |

| `20261017-180509-p3x9-freeinf` | `python -I code/p3x/x9_freeinf.py --out results/p3_explore/x9 --rundir <run> --procs 4` | FreeInference 版 T1／T2／T3（§2.4） | 被下一列覆蓋（`results/` 的 `fi_*.csv` 已換成下一列的；原始輸出留在 run 目錄的 `stdout.log`） | 0 |
| `20261017-181049-p3x9-freeinf` | 同上（加了只報告的「死掉 ÷ 新寫入」欄，§6 第 12 點） | FreeInference 版 | `fi_t1_census.csv`、`fi_t1_labels.csv`、`fi_t2_shared_prefix.csv`、`fi_t3_window_censoring.csv`；`fi_gaps_s.npy`、`fi_shares.npy` 在 run 目錄 | 0 |

（`x9_crosscheck.py` 要 import `m7_sim`，所以沒有用 `-I`；它不讀任何下載的資料。兩次 freeinf 的已登記欄位逐位相同：兩次的 stdout 數字一致。）

---

## 4. 結果

### 4.1 想法總表（20 個）

判定照 §2.1。T1–T3 的判定照 §2.2 字面（§4.2–4.4）。標記：〔卡片轉引〕＝repo 裡其他 agent 讀原文寫的卡片，我沒有自己讀原文；〔搜尋摘要〕＝只看到搜尋結果的摘要，沒有開原頁。

| # | 想法（一句話） | 為什麼在延後＋Cake 下才有意思 | 前作（來源、讀到哪） | 資料拿不拿得到 | 最便宜的測法 | 判定＋一句理由 |
|:--|:--|:--|:--|:--|:--|:--|
| I1 | **思考內容剝除**：思考模型下一輪會把上一輪的推理拿掉，剛寫進去的輸出 KV 立刻死掉 | 延後設計把 decode 出來的 KV 全寫進 CPU；死掉的正好是最新寫的，LRU 會留它最久 | Lachesis：reasoning 占一輪寫入 66–83%、只活一輪；HBM／HBF 模擬，沒和延後版比〔卡片轉引 `A_Lachesis.md`：原文 p.2、p.9〕。Qwen 維護者：前幾輪的思考內容應移除，官方 template 會自動做〔文件：QwenLM/Qwen3 discussion #1398〕。Anthropic：舊輪的 thinking block 被剝掉（較新模型保留）〔搜尋摘要：docs.anthropic.com context-windows、extended-thinking〕。CommitKV、AgentKV 是 GPU 內有損壓縮〔摘要：arXiv 2608.07855；標題：2609.14872〕 | Bailian thinking（1,200 條父子邊、509 個回來的 session）、coder〔實測〕 | **T1（已做）** | **可能**：只有思考 trace 有量（5.30%，剛過 5%），而且回來的 session <1,000；死掉是永久的，延後版只要被告知就看得到，是提示 API 問題，前作 Lachesis 已做 |
| I2 | **重新生成／編輯舊訊息**（分岔） | 重新生成讓舊分支的尾段死掉；編輯讓編輯點之後全部死掉；使用者接哪一支要猜 | SGLang RadixAttention 用樹狀快取處理分支〔搜尋摘要：arXiv 2312.07104〕。重新生成的頻率：查不到 | Bailian 四個 trace **0 個兄弟**〔實測 `t1_census.csv` fork_rate＝0〕；事後檢查：別的根 session 有同樣長歷史的回來請求只有 0.6–6.3%，不像是把分岔記成新 session〔實測 `20261017-175449-p3x9-forkcheck`〕。沒有其他公開資料有重新生成事件（查不到） | 要有帶重新生成事件的 trace | **死路**：資料拿不到（Bailian 一個兄弟都沒有，是真的沒有還是公開版處理掉的，查不到） |
| I3 | **前端截斷／agent 壓縮（compaction）** | 框架在固定門檻（例如 context 快滿）時改寫歷史，舊 KV 幾乎全死；這是框架事先知道的事，可以避免把快要被壓縮的 session 寫進慢層 | Claude Code 的 compaction 設計成重用原本的 cache 前綴〔二手：implicator.ai、第三方原始碼分析；非官方〕；Practical Online KV Cache Compaction for LLM Agents〔標題：arXiv 2608.00902，未讀〕 | Bailian REWRITE 類只占邊的 0–3.7%〔實測〕；CacheWise 有 `cache_creation_input_tokens`／`cache_read_input_tokens`，只有 94 個 session〔資料欄位〕。**登記後**：FreeInference 有作者標的 `cause=compaction`，這些邊只占 ΣS 的 0.04%、死掉質量的 2.4%〔實測 `20261017-181049-p3x9-freeinf`，`fi_t1_labels.csv`〕 | 在 FreeInference 依 `cause` 數 | **可能**（窄）：是框架提示問題、不是預測。FreeInference 版偏死路（壓縮很少見） |
| I4 | **跨 session 共用的前綴**（system prompt、熱門文件） | session 層級降級會把熱門 block 跟著冷 session 搬下去；Cake 的重算線若共用前段還在 GPU，就從更深、更貴的位置開始 | KVCache-in-the-wild：跨使用者命中很少（Fig. 6）、To-B 97% 的命中來自單輪請求〔卡片轉引 E05：原文 p.2–5〕；RAGCache：知識樹＋前綴感知替換〔搜尋摘要：arXiv 2404.12457〕；KVFlow：樹狀快取的節點級淘汰〔搜尋摘要：arXiv 2507.07400，NeurIPS'25〕 | Bailian 四個 trace、Mooncake〔實測〕 | **T2（已做）** | **可能**：共用占比 14.8–20.6%；不需要預測（降級當下看得到 ref count），會預測的部分（文件會不會再熱）前作多 |
| I5 | **trace 視窗截斷**：1–2 小時 trace 把晚點回來當成不回來 | O-丟掉、O-不寫慢層的價值全靠「誰不回來」 | 查不到專門談 KV oracle 視窗設限偏誤的論文；in-the-wild：trace A 的 block 90% 在 612 s 後不再被重用〔卡片轉引 E05：原文 p.8〕；第三階段計畫 §9 第 3 條已列「尾端被誤當不回來」（講的是訓練標籤），但 X8 §2.1 寫「trace 結束前不回來＝永遠不回來」，X5 §2 的 oracle「不回來的直接丟」 | WildChat-1M（每個 assistant 訊息有時間戳，一年多）〔實測〕 | **T3（已做）** | **可能**：M_1h＝14.6%（介於 5% 與 20%）；不是新研究問題，是 X1／X5／X8 的有效性問題 |
| I6 | **日夜週期** | 延後設計需要閒置時間；低谷有、尖峰可能沒有；負載本身很好預測 | BurstGPT：對話型流量白天高、晚上低，API 型沒有週期〔搜尋摘要：arXiv 2401.17644〕；in-the-wild：同一時段的重用分布跨 5 個工作日相近〔卡片轉引 E05：原文 p.18〕 | Azure LLM 2024（一週，只有到達與長度）、BurstGPT（本機有）；**沒有 session** | 尖峰時段的持續長度 × KV 產生速率 ÷ SSD 寫頻寬 | **可能**（窄）：單卡的併發模擬 ρ_KV 最大 0.99（main.tex §4.5，模擬），只有多卡共用 SSD 的尖峰才有事，那裡我們量不到 |
| I7 | **To-B API 的週期性呼叫** | 程式化的客戶端回來時間幾乎確定 | in-the-wild：To-B 的重用 80% 在 10 s 內〔卡片轉引 E05：原文 p.6〕；BurstGPT：API 沒有週期〔搜尋摘要〕 | Bailian traceB：172,800 個請求**全是單輪**（0 條父子邊）〔實測〕；session 層級工作集是不重複 block 的 2.60 倍（共用 system prompt）〔實測 `20261017-175022-p3x9-extra`〕 | — | **死路**：重用在幾秒內，KV 還在 GPU／CPU，不碰降級或慢層 |
| I8 | **客戶端明確宣告**（cache_control、TTL、prompt_cache_key） | 宣告的 TTL 直接告訴系統什麼時候可以丟、要不要寫慢層，取代預測 | Anthropic prompt caching 的 5 分鐘／1 小時（CacheWise 的 usage 欄有 `ephemeral_5m_input_tokens`）〔資料欄位〕；OpenAI `prompt_cache_retention`（24h）、新模型改 `prompt_cache_options.ttl`〔搜尋摘要：developers.openai.com prompt-caching〕；KVFlow 用 agent step graph 導引淘汰與預取〔搜尋摘要〕；Continuum 用工具時長設 TTL〔卡片 `D8_Continuum.md`〕；X6 的 P-hint 分類 | CacheWise 94 個 session；沒有大規模帶宣告的 trace | 在 CacheWise 數「間隔 >5 分鐘後的 cache_creation」 | **可能**：真正能拿掉預測，但這是設計點；X6 在查真實系統的旋鈕；沒有 ≥1,000 session 的資料 |
| I9 | **MCP 工具中繼資料** | 若工具宣告時長或副作用，就知道 agent 多久回來 | MCP tool annotations：readOnlyHint、destructiveHint、idempotentHint、openWorldHint，都是建議性質〔二手：SDK 文件；官方 spec 頁沒開〕 | 無 | — | **死路**：現有欄位沒有時長或 context 壽命；時長由 X3／X8 用工具名的歷史處理 |
| I10 | **多模態（圖片 token）** | 圖片 block 重算要跑視覺編碼器，f(i) 不再隨位置平滑，Cake 會合點跟著變 | D5：多存一份圖片編碼只多 22% 空間＝寫穿〔repo `research_20261010_directions/SUMMARY.md` §2〕；MPIC〔卡片 `A_MPIC.md`〕 | Bailian traceA `type=image` 只有 135 個有子的父，死掉的 KV 為 0〔實測 `t1_by_type.csv`〕 | — | **死路**：D5 已判；是讀取時決定，不需要預測；占比小 |
| I11 | **多個 session 同時回來的還原順序**（短的先） | 還原互搶 GPU 重算與 I/O | D8-15（跨請求 Cake）、Strata〔卡片 `D8_Strata_排程補充.md`〕；X4 | Mooncake／Bailian 到達時間（X4 在用） | — | **死路**：還原大小與所在層在讀取時都知道，不需要預測；X4、D8-15 涵蓋 |
| I12 | **打字或開啟對話頁時推測性還原** | 前端事件比請求到達早，給慢層預取提前量 | X3（打字時間）；CachedAttention 的佇列預取〔卡片 `A_CachedAttention_補充.md`〕；UI 事件（開分頁、focus）的資料：查不到 | 無公開 UI 事件資料 | X3 的提前量算術 | **可能**：取決於 X3；沒有資料就不能變有看頭 |
| I13 | **付費等級／SLO 不同** | 依等級加權降級與還原 | CASCADE：每個請求的 SLO 剩餘預算同時驅動排程與「從深層還原／預取、留在 HBM、或重算」〔摘要：arXiv 2608.06557〕；ProServe：依優先序卸載〔搜尋摘要：arXiv 2512.12928〕；Tail-Optimized LRU〔卡片 `D8_TailOptLRU.md`〕 | 公開 trace 沒有付費等級（Bailian A／B 是不同服務） | — | **死路**：等級在決定當下就知道，不是預測；2026 年已有 CASCADE；沒有資料 |
| I14 | **用電或用錢當目標**（含預取猜錯的浪費） | 目標換了，猜錯的代價也換了：多搬一份沒人用的資料，在能耗帳上比在時間帳上貴 | D8-01／02；main.tex §6.5 已量：CPU 與本地 SSD 層「只載入」比 Cake 省 62–70% GPU 能量（一個 seed）；MatKV〔卡片 `D8_MatKV.md`〕 | MI300X 能量計數器（3090 沒有，CLAUDE.md §3）；價格要從公開頁抄 | 把 X1 的預取位元組換成焦耳／美元 | **可能**：要等 X1 的預取 headroom；能耗只能在 MI300X 量 |
| I15 | **結束語（「謝謝」等）當成不回來的訊號** | 回合結束當下用文字判斷「不會回來」，就不寫慢層 | LPC：用對話內容預測會不會接續〔搜尋摘要：NeurIPS'25〕；SAECache：token 級〔搜尋摘要：arXiv 2605.18825〕 | WildChat 有文字 | 固定關鍵字規則在 WildChat 最後一輪 vs 非最後一輪的命中率 | **可能**：有前作；價值上限是 X1 的 O-不寫（D4：只有 CPU ≤10% 工作集才有差） |
| I16 | **模型更新讓 KV 全部失效** | 換權重後所有存的 KV 一次作廢 | 查不到（RL 的過期 KV 見 D8-07） | 無 | — | **死路**：罕見、全域、事件本身就知道，不是預測問題 |
| I17 | **重試／重複請求** | 同一個 prompt 幾秒內再送一次 | 查不到 | Bailian：子和父輸入完全相同的比例 0、輸入相同的兄弟 0〔實測 `t1_census.csv`〕 | — | **死路**：資料裡沒有；就算有也是 GPU 命中 |
| I18 | **搜尋型請求每輪換掉檢索內容** | 中間的檢索段被換掉，後段全部失效 | in-the-wild：file／search 幾乎沒有空間局部性〔卡片轉引 E05：原文 p.7〕 | Bailian traceA `type=search`：3,763 個有子的父，死掉的 KV 為 0〔實測 `t1_by_type.csv`〕 | — | **死路**：生產資料顯示檢索內容沒有被換掉 |
| I19 | **prefix 樹分支點的預測** | 預測新請求會從哪裡分岔，分岔點之前的留著 | KVFlow、SGLang〔搜尋摘要〕 | 沒有分岔資料（I2） | — | **死路**：資料拿不到 |
| I20（看到 FreeInference 結果後才加） | **agent 中途改 system prompt／工具定義，整段 KV 失效** | FreeInference 的輸入依「工具定義 → system prompt → 訊息」排列〔文件：資料集 README〕，前面一改，後面全部要重算；延後設計存的整段都死掉；會不會「切回」舊的那段（`transition=switch`）要猜 | Anthropic：改 thinking 參數會讓含訊息的快取前綴失效，system prompt 與工具定義的快取不受影響〔搜尋摘要：Bedrock 文件〕；Claude Code 團隊把快取命中下降當事故、設計成不改前綴〔二手：implicator.ai〕 | FreeInference：死掉質量的 37%（system_prompt）＋19%（tool_definitions）〔實測 `fi_t1_labels.csv`〕 | 已有（T1-FI 的分類） | **可能**（窄）：引擎在下一個請求到時就看得到前綴分岔，不用預測；只有「會不會切回」要猜。但它讓「session 只會往後長」的模擬假設在 agent trace 上不成立 |

### 4.2 T1：回合結束就死掉的 KV

〔實測，run `20261017-174829-p3x9-census`，`results/p3_explore/x9/t1_census.csv`、`t1_by_type.csv`〕

規則 6 斷言：`len(hash_ids)==ceil(input_length/16)` 四個 trace 都是 1.000；父子邊的 timestamp 順序、`turn` 加一都是 1.000（traceB 沒有父子邊，記 NaN）。

| trace | 請求 | 根 session | 有子的父（＝回來的邊） | 這些父所屬的 session | APPEND／STRIP／REWRITE（邊的比例） | **ΣD⁻/ΣS**（下界） | ΣD⁺/ΣS（上界） | 分岔率 |
|:--|--:|--:|--:|--:|:--|--:|--:|--:|
| traceA（To-C） | 43,058 | 23,101 | 19,957 | 9,012 | 100%／0／0 | **0.00%** | 0.00% | 0 |
| traceB（To-B） | 172,800 | 172,800 | 0 | 0 | —（全是單輪） | — | — | — |
| thinking | 10,812 | 9,612 | 1,200 | **509** | 72.0%／24.3%／3.7% | **5.30%** | 9.90% | 0 |
| coder | 43,011 | 26,406 | 16,605 | 6,693 | 98.6%／0.99%／0.43% | **0.05%** | 0.17% | 0 |

- **照字面判定：可能**。有看頭要「≥1,000 個有子的父、≥1,000 個獨立 session、≥20%」：thinking 有 1,200 個父但只有 509 個 session，而且只有 5.30%。死路要「四個都 <5% 而且分岔率都 <5%」：thinking 是 5.30%，不是 <5%。
- thinking 的死掉質量有 89.0% 在 STRIP 類（coder 83.0%）。
- 只報告〔實測 run `20261017-175022-p3x9-extra`，`t1_breakdown.csv`〕：
  - 改用「父這一輪寫的輸出」當分母，thinking 死掉的是 **28.2%（下界）到 52.6%（上界）**；coder 0.35%–1.16%；traceA 0。
  - thinking 的 STRIP 邊：父輸出中位數 1,441 token，下一輪輸入只多了 719 token（中位數）。APPEND 邊多出來的 token（Δ − 父輸出）中位數 64.5。
  - traceA 的 APPEND 邊多出來的 token 中位數 23（p10 14、p90 201），像是 template 加一句短的追問〔判讀〕。traceA 100% 是 APPEND、0 個兄弟，是不是公開版的構造造成的，從資料本身分辨不出來（§6 第 4 點）。
- Mooncake（只報告，用前綴鏈重建父子，不是明確欄位）：conversation 3,974 條邊中 STRIP 10.7%，toolagent 16,399 條中 21.7%（REWRITE 依重建規則必為 0）。這些 STRIP 可能是重建時接錯父，不能解讀成剝除〔判讀〕。
- 〔判讀〕thinking 有 72% 的邊把整段輸出帶到下一輪，跟 Qwen 官方「前幾輪的思考內容應移除」不一致。Bailian 的上游 FAQ 說應用端只剝 `<think>`／`</think>` 兩個特殊 token。若是這樣，輸出的 token 幾乎都還在，但開頭少了一個 token，後面的位置全部錯開，輸出 KV 仍然不能當前綴用。這種情況會被算成 APPEND，所以 D⁺ 可能還低估了。trace 沒有公開輸出的 hash，這一點量不到（NOT_MEASURED）。

### 4.3 T2：跨 session 共用前綴

〔實測，run `20261017-174829-p3x9-census`，`results/p3_explore/x9/t2_shared_prefix.csv`；算術用 MI300X 的 f(i)＝`calib_c1.csv`（run `20261008-130316-m7-c1`）、block ≥80 線性外插（slope 0.8877 ms／block）〕

| trace | 回來請求 | **位元組加權共用占比** | 中位數／p90 | 共用 ≥50% 的回來比例 | 前綴中位數（token） | 共用中位數（token） | Cake 總時間省下（b\*=32／NFS）〔算術〕 |
|:--|--:|--:|:--|--:|--:|--:|:--|
| traceA | 19,957 | **14.8%** | 0.20／0.94 | 34.5% | 1,168 | 64 | 6.8%／11.5% |
| thinking | 1,197 | **17.0%** | 0.10／0.98 | 27.9% | 5,104 | 320 | 10.9%／10.5% |
| coder | 16,602 | **20.6%** | 0.053／0.81 | 18.0% | 3,488 | 112 | 14.7%／13.7% |
| traceB | 0 | — | — | — | — | — | — |
| Mooncake conversation（只報告） | 3,974 | 9.8% | 0.09／0.50 | 13.7% | 6,656 | 512 | 8.3%／8.3% |
| Mooncake toolagent（只報告） | 16,399 | 64.1% | 1.0／1.0 | 80.4% | 6,144 | 6,144 | 60.3%／58.2% |

- **照字面判定：可能**（三個 Bailian trace 都在 5% 與 30% 之間）。
- 「Cake 省下」＝假設共用前綴的整個 512-token chunk 已在 GPU（免費），Cake 只還原其餘部分。三個 Bailian trace 每個回來請求的中位數都是 0（大部分請求共用不到一個完整 chunk），所以看總和。Cake 函式和 `m7_sim.sim_restore` 逐位一致（run `20261017-175709-p3x9-xcheck`，最大相對差 3.9×10⁻¹⁶）。
- 只報告〔實測 run `20261017-175022-p3x9-extra`，`t2_session_vs_block_workingset.csv`〕：Σ（每個 session 最長請求的 block 數）÷ 不重複 block 數＝traceA **1.27**、traceB **2.60**、thinking **1.58**、coder **2.04**。換成 Llama-3.1-8B BF16（128 KiB／token）：traceA 是 6,263 GiB 對 4,942 GiB，coder 是 20,751 GiB 對 10,161 GiB。
- 只報告（事後）〔實測 run `20261017-175449-p3x9-forkcheck`，`t1t2_forkcheck.csv`〕：根請求（第 1 輪）和更早的另一個根共用 ≥1,024 token 前綴的比例：traceA 4.9%、thinking 43.0%、coder 33.2%、traceB 10.7%。共用主要是長的 system prompt 或範本，不是複製出來的對話歷史（另一個 session 有同樣長歷史的回來請求只有 0.6–6.3%）。

### 4.4 T3：視窗截斷

〔實測，run `20261017-174829-p3x9-window`，`results/p3_explore/x9/t3_window_censoring.csv`、`t3_gap_distribution.csv`〕

規則 6 斷言：頂層 `turn`＝user 訊息數 1.000；時間戳不遞減 1.000；頂層 `timestamp`＝最後一個訊息 1.000。**user 訊息沒有時間戳（0／1,960,074），assistant 全有（1,960,074／1,960,074）**，所以照 §2.2 用「下一個 assistant 訊息」當回來時刻（間隔含下一輪的生成時間）。

| 量 | 值 |
|:--|--:|
| 對話數／回合結束數／有回來的間隔數 | 837,989／1,960,074／1,122,085 |
| **M_1h**（1 小時視窗標「不回來」的裡面其實有回來的比例） | **14.6%** |
| M_2h | 9.8% |
| M_1h／M_2h，只算 ≥2 輪的對話（只報告） | 30.2%／21.5% |
| 回來間隔中位數／p75／p90／p99 | 128 s／368 s／1,253 s／27,511 s |
| 間隔 >5 min／>10 min／>1 h／>2 h／>24 h | 28.9%／17.6%／4.47%／2.65%／0.31% |

- **照字面判定：可能**（5% ≤ 14.6% < 20%）。
- 一致性（事後、只報告）〔實測 run `20261017-175327-p3x9-gapcheck`，`t3_gap_consistency.csv`〕：只看 1 小時內回來的間隔，WildChat 的中位數 119 s、p90 812 s；Bailian traceA（到達到到達）111 s、717 s；coder 116 s、882 s；thinking 29 s、251 s。WildChat 在視窗內的部分和生產 To-C chat 很像，所以拿它推視窗外的尾巴不算離譜〔判讀〕；但這不能證明視窗外的尾巴也像。
- 〔判讀〕對 X1／X5／X8：如果 Mooncake 的使用者回來的樣子和 WildChat 一樣，在 1 小時的 Mooncake 上被 oracle 當成「永遠不回來」而丟掉的回合結束，大約每 7 個有 1 個（只看多輪對話是每 3 個有 1 個）在現實中會回來。這會讓 O-丟掉、O-不寫慢層看起來比實際好，所以 P3-3 偏向 GO（假陽性的方向）。

### 4.5 FreeInference 版（登記之後才加，§2.4）

〔實測，run `20261017-181049-p3x9-freeinf`，`results/p3_explore/x9/fi_t1_census.csv`、`fi_t1_labels.csv`、`fi_t2_shared_prefix.csv`、`fi_t3_window_censoring.csv`〕

規則 6 斷言：(a) 16·`len(block_ids)` ÷ `role_tokens` 加總落在 [0.9, 1.1]：1,222,199／1,222,199（1.000）；(b) 作者標 append 的相鄰請求，前綴延續 1.000；(c) `step` 遞增 1.000、`start_ms` 不遞減 0.999999；二分搜尋的前提（鏈式 block）每週抽 2,000 條邊和逐一比對的結果一致 1.000。
規模：16 週、1,222,199 個請求、14,036 個 session（含 2,034 個 subagent session；README 寫的 12,002 是不含 subagent 的數）、1,208,163 條相鄰請求邊、13,954 個 session 至少有一條邊。

**T1-FI（判定：死路）**

| 量 | 值 |
|:--|--:|
| 邊的分類 APPEND／STRIP／REWRITE | 80.1%／16.0%／3.9% |
| **ΣD⁻/ΣS**（下界）／ΣD⁺/ΣS（上界） | **1.49%**／1.54% |
| 只報告：ΣD⁻ ÷ Σ新寫入（每一步超出和前一步共同前綴的輸入＋輸出）／上界 | **57.0%**／58.9% |

- **照字面判定：死路**（1.49% < 5%）。
- 死掉質量依作者的標籤（`fi_t1_labels.csv`）：mutation｜system_prompt 36.9%、mutation｜tool_definitions 18.7%、mutation｜other 11.4%、mutation｜dropped_turns 9.8%、switch 8.7%、mutation｜injection 7.6%、mutation｜tool_result 3.6%、mutation｜compaction 2.4%、append 0.9%。被標成 append 的邊占 ΣS 的 96.2%。
- 我的分類 vs 作者標籤（邊數）：我的 STRIP 裡 191,905 條作者標 append。這是因為 `completion_tokens` 是供應商的 tokenizer、Δ 是 o200k，加上工具呼叫的輸出在下一輪會被重新排版；這些邊的死掉質量只占 0.9%，所以不影響結論，但 STRIP 這一類在 FreeInference 上不能解讀成「剝除」〔判讀〕。REWRITE 的 47,244 條裡 44,979 條作者標 mutation、2,265 條標 switch，兩邊一致。
- 〔判讀〕兩個分母差很多：每一步都重送整段 context，ΣS 把同一段前綴算了很多次；「新寫入」才是真的寫進 CPU 的量。照新寫入算，**agent session 寫進去的 KV 有一半以上在下一步就不再是前綴**，主要原因是中途改 system prompt 或工具定義（工具定義排在最前面）。這些在下一個請求到達時引擎就看得到（前綴分岔），不需要預測；需要猜的只有「會不會切回舊的那段」。

**T2-FI（判定：可能）**

| 量 | 值 |
|:--|--:|
| 回來請求（有前一步、共同前綴 >0） | 1,206,329 |
| **位元組加權共用占比**（≥2 個 session） | **25.7%** |
| 跨使用者版本（≥2 個 `user_id`） | **1.7%** |
| 不含 subagent session | 25.7% |
| 中位數／p90／共用 ≥50% 的比例 | 0.13／0.98／21.8% |
| session ÷ 不重複 block（16 週加總） | 1.17 |

- **照字面判定：可能**（5% ≤ 25.7% < 30%）。
- 〔判讀〕共用幾乎都在同一個使用者的不同 session 之間（同一套工具定義、system prompt、同一個 repo），跨使用者只有 1.7%。這和 in-the-wild「跨使用者命中很少」一致。資料集是不是對每個使用者加不同的鹽，README 沒寫（查不到）；如果有，跨使用者的數字會被壓低。

**T3-FI（判定：有看頭）**

| 版本 | M_1h | M_2h | 間隔中位數／p90 | 間隔 >1 h | 間隔 >24 h |
|:--|--:|--:|:--|--:|--:|
| 全部請求（含工具回來） | **47.5%** | 35.6% | 1.9 s／34 s | 0.30% | 0.008% |
| 只算下一步由使用者觸發 | 36.8% | 27.3% | 96 s／1,040 s | 4.17% | 0.11% |

- **照字面判定：有看頭**（47.5% ≥ 20%）。只看人回來的版本也是 36.8%。
- 被丟掉的負間隔（下一步的開始早於上一步的結束）：17 條，已截成 0。
- 〔判讀〕agent session 很長（平均約 87 步），1 小時的視窗切下去，大部分正在進行的 session 都會被切斷。所以「視窗內沒回來＝不回來」在 agent trace 上錯得最多。session 不跨週，所以這仍是下界。

---

## 5. 前作（本次查證到的程度）

| 前作 | 讀到哪 | 和 X9 哪一列有關 |
|:--|:--|:--|
| Lachesis（arXiv 2610.08378） | 〔卡片轉引〕`docs/research_20261009_explore/cards/A_Lachesis.md`（該卡讀了全文 p.1–11） | I1 |
| Qwen3 multi-turn thinking 移除 | 〔文件〕https://github.com/QwenLM/Qwen3/discussions/1398 | I1 |
| Anthropic extended thinking／context windows | 〔搜尋摘要〕https://docs.anthropic.com/en/docs/build-with-claude/context-windows 、https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking（沒有開原頁） | I1 |
| particula.tech 部落格：Qwen3／gpt-oss template 破壞前綴快取 | 〔二手，搜尋摘要〕https://particula.tech/blog/vllm-prefix-cache-miss-multi-turn-agents | I1 |
| CommitKV（arXiv 2608.07855） | 〔摘要〕https://arxiv.org/abs/2608.07855 （GPU 內有損壓縮，不是分層） | I1 |
| SGLang／RadixAttention（arXiv 2312.07104） | 〔摘要〕 | I2、I19 |
| KVCache Cache in the Wild（ATC'25，arXiv 2506.02634） | 〔卡片轉引〕`docs/research_20261006_eval/cards/E05_eviction_policy.md`（該卡讀了 v5 全文） | I4、I5、I6、I7、I18 |
| RAGCache（arXiv 2404.12457） | 〔搜尋摘要〕venue 未查證 | I4 |
| KVFlow（arXiv 2507.07400，NeurIPS'25） | 〔搜尋摘要〕 | I4、I8、I19 |
| BurstGPT（arXiv 2401.17644） | 〔搜尋摘要〕 | I6、I7 |
| OpenAI prompt caching | 〔搜尋摘要〕https://developers.openai.com/api/docs/guides/prompt-caching | I8 |
| MCP tool annotations | 〔二手：SDK 文件〕官方 spec 頁沒開 | I9 |
| CASCADE（arXiv 2608.06557） | 〔摘要〕https://arxiv.org/abs/2608.06557 | I13 |
| ProServe（arXiv 2512.12928） | 〔搜尋摘要〕 | I13 |
| LPC（NeurIPS'25） | 〔搜尋摘要〕 | I15 |
| SAECache（arXiv 2605.18825） | 〔搜尋摘要〕 | I15 |
| Continuum、Tail-Optimized LRU、MatKV、Strata、MPIC、CachedAttention | 〔卡片〕repo 內既有卡片（其他 agent 寫的） | I8、I13、I14、I11、I10、I12 |
| FreeInference agentic trace（HF `harvardMadsys/freeinference_agentic_trace`；論文 *From Requests to Sessions*） | 〔文件〕資料集 README（本機 `datasets/freeinference_agentic_trace/README.md`）；論文沒讀 | T1-FI–T3-FI、I3、I20 |
| Bedrock 的 Claude extended thinking 文件（改 thinking 參數讓含訊息的快取失效） | 〔搜尋摘要〕https://docs.aws.eu/bedrock/latest/userguide/claude-messages-extended-thinking.html | I20 |

- 沒有下載任何 PDF（`/ssd8/hungwei/paper-hkv/tmp/p3x9/` 是空的）；所有論文層級的引用都是摘要或卡片轉引，不是我讀的原文。
- 「回來時間預測」的文獻由另一個 agent 在 `docs/phase3_20261017/research/` 做，這裡沒有重做。

---

## 6. 偏離與問題（照實列）

1. **第一次 census 失敗**（run `20261017-174657-p3x9-census`，exit 1）：`AssertionError: …qwen_traceB_blksz_16.jsonl: 子的 timestamp ≥ 父 只有 0.0000`。原因是我的程式：traceB 沒有任何父子邊，比例算成 0／max(1,0)＝0，觸發斷言。改成「沒有邊就記 NaN、不做 (b) 斷言」後重跑（run `20261017-174829-p3x9-census`）。**修改時我已經在失敗那次的 stdout 看到 traceA 的 T1／T2 數字**；定義與門檻都沒有改。失敗 run 的目錄原樣保留。
2. **事後加的分析**（看到 T1–T3 結果之後才寫）：`x9_extra.py`（改用父輸出當分母、session vs block 工作集）、`x9_gapcheck.py`（WildChat 與 Bailian 在 1 小時內的間隔比較，09:53:11Z 開始寫）、`x9_forkcheck.py`（分岔是不是記成新 session）、`x9_crosscheck.py`。全部只報告，不改判定。
3. **T1 的 thinking trace 剛好過 5%**（5.30%），判定落在「可能」而不是「死路」，差距很小；而且它的回來 session 只有 509 個，就算 ≥20% 也不能判有看頭。
4. **Bailian 沒有任何兄弟**：重新生成、編輯在這份資料裡完全看不到。真實服務不太可能沒有重新生成〔判讀〕，可能是公開版的取樣或匿名化把它處理掉了；上游 README 和 FAQ 沒有說明（查不到）。所以 I2、I19 判死路是「資料拿不到」，不是「現象不存在」。
5. **WildChat 的 user 訊息沒有時間戳**：間隔是 assistant 到 assistant，含下一輪的生成時間，所以略長於真正的使用者思考時間。WildChat 介面若不能在重新整理後接續舊對話，長間隔會被低估，M_W 是下界〔判讀，未查證介面行為〕。
6. **T2 的共用判斷用了整個 trace**（含未來的請求）：這是描述統計，不是策略。
7. **Mooncake 的父子重建**要求至少 2 個 block 的共同前綴（避免全體共用的 block 0 被當成父子）。這是在第一次跑之前改的。Mooncake 只報告。
8. **T2 的 Cake 算術**用 MI300X 的 f(i)，block ≥80 用線性外插〔外插〕；載入時間是固定的每 chunk 時間，沒有爭用與寫入干擾。只報告。
9. **前作多數只讀到摘要或搜尋摘要**（§5）。CommitKV、CASCADE 是 WebFetch 讀 arXiv 摘要頁；其他標〔搜尋摘要〕的只看過搜尋結果。
10. **FreeInference 是登記之後才加的資料**（README §4，10:01Z）。§2.4 的判準在 10:02:37Z–10:03:00Z 寫，那時已經知道 Bailian／WildChat 的結果（門檻照抄 §2.2，沒有改）。照 README，總結以原登記版為準。
11. **T1 的分母在 agent trace 上不合適**（看到 FreeInference 結果後才發現）：ΣS 是每一步的「整段輸入＋輸出」加總，agent 每一步都重送整段 context，同一段前綴被算了很多次，所以 ΣD⁻/ΣS（1.49%）把死掉的比例稀釋了。這個分母是我在 §2.2 為 chat trace 定的，沒有預先想到 agent 的情況。判定照字面仍是死路；另外報「死掉 ÷ 新寫入」（57.0%）當只報告的數字，不改判定。Bailian 也有同樣的問題（只是程度輕），所以 Bailian 的 T1 另報了「÷ 父的輸出」。
12. **FreeInference 跑了兩次**：`20261017-180509-p3x9-freeinf`（原版）與 `20261017-181049-p3x9-freeinf`（只多加第 11 點的欄位）。兩次的已登記數字逐位相同（兩個 stdout 的逐週行與規則 6 行完全一樣，`cmp` 檢查過）；`results/` 裡是第二次的。
13. **FreeInference 的 T1 用了兩種 tokenizer**：輸入長度用 o200k 的 block 數，輸出用供應商回報的 `completion_tokens`。這讓 191,905 條作者標 append 的邊被我分到 STRIP；它們的死掉質量只占 0.9%（§4.5）。
14. **FreeInference 的 session 數**：14,036 個含 2,034 個 subagent session。我把 subagent session 也當獨立 session；不含它們也遠超過 1,000，判定不受影響。T2 另報了不含 subagent 的版本（25.7%，一樣）。
15. **I20 是看到 FreeInference 結果之後才加的想法**，不在原本的清單裡。
16. 時間：約 09:37Z 開始、10:20Z 寫完（約 45 分鐘），沒有超過 3 小時預算。

---

## 7. 如果要繼續，下一步

1. **（最值得做）T3 → X1／X5／X8／P3-3 的設限保護**〔判讀〕：在 oracle 與對手都只評估「離 trace 結束還有 ≥H 秒」的降級／丟掉決定（H＝1,800 s 與 3,600 s；Mooncake 只有 58.9 分鐘，所以要改用 Bailian 2 小時或 FreeInference 一週，或把 trace 前段當評估、後段只當「回不回來」的觀察窗）。比較加保護前後的 O-丟掉、O-不寫慢層 headroom。若 headroom 掉超過一半，P3-3 的判定要用加保護的版本。這要先寫進 P3-3 的事先登記。FreeInference 一週的 session 本來就被週界切斷，所以也要對「週末前 H 秒」做同樣的處理。
2. **T2 → 模擬器的單位**：X1／X5／X8 的 CPU 容量若寫「占工作集 %」，要註明是 session 加總還是不重複 block；兩者差 1.27–2.60 倍。若模擬是 session 層級，建議加一個 block 層級、依 ref count 的 LRU 對手（延後版、不需要預測），看 session 層級的結論會不會變。
3. **T1 → 只有在有 agent／思考模型的大型 trace 時才值得做**：要有輸出的 hash（才能分辨「整段帶過去」和「只剝了 `<think>`」）、≥1,000 個回來的 session。測法：在 X1 的模擬器裡比「LRU」和「LRU＋死掉的先丟（框架提示）」，看同樣 TTFT 下省多少 CPU 容量。這是延後版也拿得到的提示，所以只能寫成「提示 API 值多少」，不能寫成「寫入時決定」。
4. **T1-FI／I20 → agent trace 的模擬不能假設 session 只會往後長**：FreeInference 新寫入的 KV 有 57% 在下一步就不再是前綴。用 FreeInference 的方向（X1／X8 若有加）應該照作者的 `transition` 重建 session 的 KV：mutation 之後舊的那段是「沒有人指向」的 block，延後設計可以不預測就先丟（對手也要有這個機制，README §1 的對稱規則）。要猜的只剩 `switch` 會不會切回去。
5. 其他「可能」的（I3、I6、I8、I12、I14、I15）都要等 X1／X3 的 headroom：若 X1 判死路，它們的上限也跟著是零。
