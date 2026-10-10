# D4 工作負載本身的資訊值多少（一寫多讀、會不會回來）

（第 2 節在算任何策略比較之前寫死，之後沒改。第 0、1、3–7 節是結果出來後才寫的。）
（所有模擬都是「一次服務一個請求、沒有排隊」的模型，和第一階段同一個前提。標〔實測〕的是**模擬器實際跑出來的數字**，不是 GPU 量測。）

## 0. 一句話結論＋判定

**判定：有看頭，但只限「SSD 少寫」，而且是「重用預測這條規則」有看頭，不是「寫入時決定」有看頭。** TTFT 那一半是死路。

- **照第 2 節判準逐字套用**（延後版只用 §2.2 表上列的 7 個）：
  - SSD 寫入差距 gap_W 中位數 **86.8%（conversation）、89.2%（toolagent）**，超過 15% → 有看頭〔實測 20261010-060621-d4-gap-prereg〕。
  - TTFT 差距 gap_T 中位數只有 **0.015%、0.002%**（最大 0.74%）→ 這一半是死路〔同上〕。
  - 歸因規則也過：只用「寫入時知道會被讀幾次」的 WTP-a，gap_W 和 oracle 一樣〔同上〕。
- **最重要的原因（為什麼不能說「寫入時」有效）**：同一份「誰會被重用」的知識，**延到淘汰時才用**（事後加的對照組 KEVa），在 CPU 11.6 GiB/s 與 NFS 的所有格子裡，TTFT 和 SSD 寫入都**和寫入時就用（WTP-a）一模一樣**〔實測 20261010-060103-d4-gap2b、20261010-060105-d4-gap3b〕。照 10 §2，這只能寫成「重用感知的准入規則有效」，不能寫成「寫入時決定有效」。
- **兩個必須一起講的但書**：
  1. **差距大部分是「用不用 SSD」**：oracle 在 CPU ≥5% 工作集時根本不寫 SSD；而延後版裡「完全不寫 SSD 的 LRU」在 CPU 25%、50% 時也只比 oracle 慢 0.05–2.0%（CPU 11.6 GiB/s）〔實測 20261010-054330-d4-sim〕。實作時（看結果之前、但沒寫進 §2）我把這類策略也放進延後版，**放進來之後 gap_W 中位數變成 0%，判定變死路**〔實測 20261010-055356-d4-gap〕。差別只在這一點，兩個都列出來讓你判斷。
  2. **真正有差的只有 CPU 很小的時候（≤10% 工作集，約 ≤1.1 TiB）**：oracle 不寫 SSD 就和最好的延後版一樣快；延後版要嘛寫 10 TiB 到 SSD，要嘛（不用 SSD 時）慢 8.6–12.5%〔實測 d4-sim〕。沒有快的 SSD（NFS 參數）時，oracle 快 7.0–17.5%〔實測 20261010-055356-d4-gap、d4-gap3b〕。
- **真實預測器只能拿到一部分**：LPC 用對話內容預測「會不會續聊」，只補回 LRU 到 oracle 差距的 22–66%〔LPC 原文 p.23〕。

## 1. 問題是什麼、要證明什麼

- **問題**：寫 KV 的當下，如果知道
  - (a) 這個 block 之後會被讀幾次（一寫多讀，例如共用的 system prompt）；
  - (b) 這個 session 之後會不會回來；
  
  能讓 TTFT 少多少、SSD 少寫多少？
- **要證明的**：這份資訊的價值，**延後版**（看頻率、淘汰時才決定的快取策略，例如 2-hit 准入、LRU、ARC、S3-FIFO）是不是已經拿走了。
  - 「2-hit 准入」＝第 2 次被讀到才存（NVIDIA Dynamo KVBM 寫 SSD、SGLang `write_through_selective` 就是這樣）。
  - 「oracle」＝看得到未來的理想策略，當上限用。
- **如果 oracle 和最好的延後版差很少**：寫入時的工作負載資訊沒有獨特價值，這個方向是死路。

## 2. 事先寫好的判準

**寫於**：2026-10-10T05:36:38Z（`date -u`）。寫的時候只看過：三個 trace 的列數、timestamp 範圍、`len(hash_ids)==ceil(input_length/512)` 的吻合率（三個都 100%）、最長輸入。**還沒跑任何快取模擬。**

### 2.1 模型（這次怎麼算 TTFT）

- 單位：一個 block＝512 token（Mooncake `hash_ids`，載入時斷言驗證）；Llama-3.1-8B 每 block 64 MiB。
- 每個請求依 timestamp 順序處理，一次一個，不排隊、不併發（和第一階段同一個前提；D1 負責併發）。
- 一個請求的模型 TTFT＝它每個 block 的成本加總（**主模型：逐 block 相加，不重疊**）：
  - 第一次出現的 block：重算 f(i)（i＝block 在請求裡的位置；強制成本，所有策略一樣）。
  - 之前出現過的 block：在 CPU → CPU 載入時間；不在 CPU 但在 SSD → min(SSD 載入, f(i))；都不在 → f(i)。
  - f(i)：`calib_c1.csv`（run 20261008-130316-m7-c1）每個 chunk 的中位數；i ≥ 80 用線性外插〔算術〕。
  - CPU 載入：64 MiB ÷ 3.69 或 11.6 GiB/s。SSD 載入：`tier_params.json` 的 local（6.98 GiB/s）與 nfs（0.331 GiB/s）。
- GPU 層：主模型**不放跨請求快取**（GPU 只當工作區）。這會讓「第二次存取沒命中」的代價看起來最大，也就是**偏向讓寫入時資訊顯得有價值**。
- SSD 層：容量不設上限；不同策略只差「寫不寫、什麼時候寫」。
- CPU 容量：工作集（trace 裡不重複 block 的總數）的 10%、25%、50%。

### 2.2 策略

| 類別 | 策略 | 意思 |
|:--|:--|:--|
| 延後／線上（不看未來） | LRU-WT | 每個 block 都進 CPU（LRU），同時寫穿 SSD |
| | LRU-WB | 進 CPU（LRU），被趕出 CPU 時才寫 SSD |
| | 2HIT | 第 2 次被存取才准入 CPU 與 SSD（KVBM／SGLang selective 那種） |
| | 2HIT-SSD | CPU 用 LRU 全收；SSD 只寫存取 ≥2 次的 |
| | SLRU、ARC、S3-FIFO | 經典「看頻率、但在淘汰時才決定」的快取 |
| 寫入時預測（上限） | WTP-a | 寫入當下就知道這個 block 之後會被讀幾次；0 次就不收；讀完最後一次就立刻丟；其餘 LRU；SSD 只寫還會被讀、且被趕出 CPU 的 |
| | WTP-b | 寫入當下只知道「這個請求之後會不會有人接著讀」（session 會不會回來），會回來才收；LRU |
| Oracle | OPT | CPU 用 Belady（知道每個 block 下次何時被讀，可拒收）；另跑一個看成本的 Belady 變體，取兩者較好的；SSD 只寫「之後要用、那時不在 CPU、而且 SSD 載入比重算便宜」的 block |

「最好的延後版」＝每一格設定裡，所有延後／線上策略 TTFT 最低的那個。

### 2.3 判準（照 D4 任務指示，加上聚合方式）

- gap_T＝(T_延後最好 − T_OPT) ÷ T_延後最好，T 是全部請求的模型 TTFT 加總（含第一次出現的強制重算）。
- gap_W＝(W_延後 − W_OPT) ÷ W_延後。W_延後＝TTFT 在「延後最好」1.05 倍以內的延後策略中，SSD 寫入位元組最少的那個。
- 每格＝（trace, CPU 容量 3 種, CPU 頻寬 2 種），SSD＝local。每個 trace 取 6 格的**中位數**。
- **死路**：真實 trace（conversation、toolagent）兩個的中位數都 gap_T < 5% **且** gap_W < 20%。
- **有看頭**：至少一個真實 trace 的中位數 gap_T ≥ 15% **或** gap_W ≥ 15%。
- 其他＝**可能**。synthetic trace 只報告，不能單獨讓判定變成有看頭。
- **歸因規則**（回答「寫入時的工作負載資訊」本身值多少）：同樣算 WTP-a、WTP-b 對「延後最好」的 gap。若 OPT 過了「有看頭」，但 WTP-a 的 gap_T < 5% 且 gap_W < 20%，表示好處來自「知道確切的未來時間」，不是寫入時的工作負載資訊，**「寫入時資訊」這個方向最多判「可能」**。
- 敏感度（只報告，不改判定）：SSD＝nfs；Cake 重疊模型（前段重算、後段載入並行）；GPU 層放一個小 LRU。

## 3. 做了什麼

| 步驟 | 指令（都走 `m7run`） | run_id | 產出 |
|:--|:--|:--|:--|
| 載入驗證＋trace 特性 | `python -I code/m8_trace_oracle.py char` | 20261010-054225-d4-char | `results/m8_directions/d4_trace_char.csv`、`d4_reuse_dist.csv`；stack distance 存在 run 目錄 `*_stackdist.npy` |
| 補充：timestamp 解析度、session 回來 | `... char2` | 20261010-054322-d4-char2 | 同上（追加列） |
| **主模擬（事先登記）** | `... sim --cake` | 20261010-054330-d4-sim | `d4_sim.csv` |
| **主判定** | `... gap --sim d4_sim.csv --src-run 20261010-054330-d4-sim` | 20261010-055356-d4-gap | `d4_gap.csv`、`d4_verdict.csv` |
| 事後：同樣知識延到淘汰時用（KEVa、KEVb）＋GPU 層敏感度 | `... sim --cake --posthoc --gpu-blocks 0 2400 --out d4_sim2.csv` | 20261010-055332-d4-sim2 | `d4_sim2.csv` |
| 事後：CPU 2%、5% | `... sim --cake --posthoc --cpu-frac 0.02 0.05 --out d4_sim3.csv` | 20261010-055444-d4-sim3 | `d4_sim3.csv` |
| 主判定，延後版只用 §2.2 逐字列的 7 個 | `... gap --sim d4_sim.csv --src-run 20261010-054330-d4-sim --deferred-set prereg --out d4_gap_prereg7.csv ...` | 20261010-060621-d4-gap-prereg | `d4_gap_prereg7.csv`、`d4_verdict_prereg7.csv` |
| 事後判定表 | `... gap ...` | 20261010-060103-d4-gap2b、20261010-060105-d4-gap3b | `d4_gap_posthoc.csv`、`d4_verdict_posthoc.csv`、`d4_gap_smallcap.csv`、`d4_verdict_smallcap.csv` |
| 完全不重用的基準 | `... nocache` | 20261010-060157-d4-nocache | `d4_nocache.csv` |
| SCBench 結構 | `... scbench --max-tok 1000` | 20261010-054738-d4-scbench2 | `d4_scbench.csv` |
| 前作查核 | 開一個子 agent 讀原文（本機 PDF＋arXiv＋官方文件），我再抽查 6 篇的原文段落 | — | 本文 §5；抽取的文字在 scratchpad（不進 repo） |

**模擬器怎麼算**（細節見第 2 節）：
- 每個請求依時間順序、一次一個；每個 512-token block 依序存取。
- 第一次出現的 block 一定要重算（所有策略一樣）。之前出現過的：在 CPU → CPU 載入；在 SSD → min(SSD 載入, 重算)；都不在 → 重算 f(i)。
- f(i)：`calib_c1.csv` 的 80 點中位數（run 20261008-130316-m7-c1），i ≥ 80 線性外插：f(i)≈28.7＋0.888·i ms〔算術〕。
- 一個 block 64 MiB。CPU 載入 16.9 ms（3.69 GiB/s）或 5.4 ms（11.6 GiB/s）；local SSD 8.95 ms、NFS 188.8 ms（`tier_params.json`）〔算術〕。
- 所有策略都不會把 block 寫到「讀回來比重算還慢」的 SSD（位置資訊人人都有，不算工作負載資訊）。

**策略的檢查**：
- 小型隨機 trace 上 30 次：所有線上策略和 WTP-a 的命中數都 ≤ Belady〔實測，未存檔的單元測試〕。
- ARC、S3-FIFO 的容量上限在 50 個隨機 trace × 5 種容量下都沒有超過〔實測，未存檔的單元測試〕。
- 主模擬和事後模擬在共同的 1,080 列裡，TTFT、SSD 寫入、Cake 欄完全相同（可重現）〔實測 20261010-054330-d4-sim vs 20261010-055332-d4-sim2〕。

## 4. 結果

### 4.1 單位驗證（規則 6）

| 檢查 | conversation | toolagent | synthetic | 來源 |
|:--|:--|:--|:--|:--|
| `len(hash_ids)==ceil(input_length/512)` 的比例 | 100% | 100% | 100% | 〔實測 d4-char〕 |
| 同一個 hash 永遠在同一位置、前一個 hash 也相同（前綴鏈） | 0 次違反 | 0 | 0 | 〔實測 d4-char〕 |
| timestamp 範圍 | 0–3,536,999 | 0–3,536,999 | 0–1,022,025 | 〔實測 d4-char〕 |
| timestamp 是 1000 的倍數（±1）的比例 | 99.6% | 99.6% | 0.35% | 〔實測 d4-char2〕 |
| 同 session 前後兩輪的間隔 ÷ 上一輪輸出 token 數（中位數） | 344 | 339 | 479 | 〔實測 d4-char〕 |

- **判讀**：timestamp 是毫秒。
  - 若是毫秒：conversation 長 58.9 分鐘；兩輪間隔中位 126 s，等於每個輸出 token 0.34 s（含使用者思考時間），合理。
  - 若是秒：每個輸出 token 要隔 344 秒，不合理；若是微秒：解碼要每秒 2,900 token 以上，也不合理〔判讀〕。
  - 和 Mooncake 原文一致：「timestamp 0 到 3,600,000，單位毫秒」「block size 512」〔Mooncake arXiv 2407.00079 原文 p.7〕。
  - conversation、toolagent 的 timestamp 解析度約 1 秒（只有 1,180 個不同值）〔實測 d4-char2〕。
- toolagent 有 23,608 列，和 Mooncake 原文的 trace 列數相同〔原文 p.7〕〔判讀：很可能是同一份〕。

### 4.2 trace 特性（task 1）

| 指標 | conversation | toolagent | synthetic |
|:--|:--|:--|:--|
| 請求數 | 12,031 | 23,608 | 3,993 |
| 不同 block 數（工作集） | 182,790（11.2 TiB） | 183,300（11.2 TiB） | 43,924（2.7 TiB） |
| 存取中是重用的比例（容量無限時的命中上限） | 36.6% | 55.3% | 64.0% |
| 每寫幾讀（總存取 ÷ 不同 block） | 1.58 | 2.23 | 2.77 |
| **從沒被重用的 block** | **75.8%** | **78.5%** | 58.4% |
| 同上，只看前半段寫的 block（去掉 trace 截尾） | 72.6% | 76.2% | 55.9% |
| 被重用的 block 裡，**只被重用 1 次**的 | 57.7% | 55.4% | 15.9% |
| 被重用的 block 平均被重用幾次 | 2.39 | 5.73 | 4.27 |
| **2-hit 一定錯過的重用存取**（每個被重用 block 的第 2 次） | **41.8%** | 17.4% | 23.4% |
| 2-hit 的學習延遲：第 1→2 次存取的時間，中位（p90） | 147 s（621 s） | 123 s（477 s） | 153 s（454 s） |
| 重用距離（時間），中位（p90） | 114 s（519 s） | 0 s（243 s） | 59 s（282 s） |
| 重用距離（中間讀過多少不同資料），中位（p90） | 522 GiB（1,974 GiB）＝工作集 4.6% | 4.5 GiB（1,003 GiB） | 392 GiB（1,275 GiB） |
| block 0 的共用 | **全部 12,031 個請求共用同一個 block 0** | 4 種 block 0，最大的占 46% | 2,211 種，55% 的請求和別人共用 |
| session 會回來的請求（之後有請求接在它最後一個完整 block 後面） | 43.9% | 44.1% | 34.2% |
| 被重用的新 block 中，所屬 session 不回來（被別的 session 分叉讀走） | 8.0% | 7.9% | 44.0% |

來源：〔實測 20261010-054225-d4-char、20261010-054322-d4-char2〕。工作集 TiB＝block 數×64 MiB〔算術〕。

- **白話**：
  - 四分之三的 block 寫了就沒人再讀。所以「全部都存」很浪費。
  - 但被重用的 block 有一半以上**只被重用一次**（〔判讀〕多半是同一個 session 的下一輪）。2-hit 准入對這種 block 一次都命中不到；conversation 有 42% 的重用存取會被 2-hit 錯過。
  - 這就是兩難：全收會被沒用的 block 擠滿；2-hit 會錯過「只回來一次」的 session。**能解開這個兩難的，只有「知道誰會回來」**——但不管是寫入時還是淘汰時知道都可以（見 4.6）。
  - 「session 會不會回來」用「任何 block 之後有人讀」來定義會失真：conversation 所有請求共用 block 0，那樣算 99.99% 都「會回來」〔實測 d4-char〕。所以改用「之後有請求接在它後面」。

### 4.3 主判定（事先登記：SSD＝local、GPU 層 0、總 TTFT、6 格中位數）

**兩種延後版集合的判定**（同一個模擬 run 20261010-054330-d4-sim）：

| 延後版集合 | conversation gap_T／gap_W 中位數 | toolagent gap_T／gap_W 中位數 | 判定 | run_id |
|:--|:--|:--|:--|:--|
| **A. §2.2 表上逐字列的 7 個**（LRU-WT、LRU-WB、2HIT、2HIT-SSD、SLRU、ARC、S3-FIFO；後三個用「淘汰時寫 SSD」） | 0.015%／**86.8%** | 0.002%／**89.2%** | **有看頭**（gap_W） | 20261010-060621-d4-gap-prereg |
| B. 實作時放進去的 25 個組合（5 種 CPU 策略 × 5 種 SSD 規則，多了「完全不寫 SSD」「趕出時讀過 ≥2 次才寫」等） | 0.015%／0% | 0.002%／0% | 死路 | 20261010-055356-d4-gap |

- B 是在跑模擬之前就寫進程式的（程式 05:41 寫好、05:43 開跑，05:51 才看到第一個結果），但**沒有寫進 §2 的文字**。照「不准事後改判準」，正式判定用 A。
- 關鍵差別〔判讀〕：B 裡有「LRU、完全不寫 SSD」（LRU＋NONE）。它在 CPU 25%、50% 時和最好延後版差不到 5%，所以「延後版也能 0 寫入」，gap_W 就變 0（B 的逐格表裡，這 4 格的「延後版在 5% 內最少寫」都是 LRU＋NONE）。
- A 裡兩個 trace 在 CPU 25%／50%＋11.6 GiB/s 的 4 格，gap_W＝100%，是因為 oracle 0 寫入、而 A 裡每個延後版都會寫 SSD——**這比的是「用不用 SSD」，不是「知不知道未來」**〔判讀〕。

以下用 B（25 個組合）的逐格表；A 的逐格值在 `d4_gap_prereg7.csv`。

| trace | 類別 | gap_T 中位數 | gap_W 中位數 | gap_T 範圍 | gap_W 範圍 |
|:--|:--|:--|:--|:--|:--|
| conversation | OPT | 0.015% | 0% | 0–0.74% | 0–100% |
| conversation | WTP-a | 0.015% | 0% | 0–0.74% | 0–100% |
| conversation | WTP-b | −2.9% | 0% | −5.9–0.53% | −213–1.4% |
| toolagent | OPT | 0.002% | 0% | 0–0.54% | 0–100% |
| toolagent | WTP-a | 0.002% | 0% | 0–0.54% | 0–100% |
| toolagent | WTP-b | −2.7% | −123% | −5.3–0.39% | −367–9.6% |
| synthetic（只報告） | OPT | 0.27% | 65.6% | 0–3.4% | 56–100% |

來源：〔實測 20261010-055356-d4-gap，`d4_verdict.csv`〕。負數＝比最好的延後版還差。

- **用 B 的話**：兩個真實 trace 的中位數都 gap_T < 5% 且 gap_W < 20% → 死路。用 A 就是上面的「有看頭（gap_W）」。
- **為什麼 TTFT 幾乎沒差**：
  - 總 TTFT 裡大部分是「第一次出現、一定要重算」的 block：conversation 佔 oracle TTFT 的 95%，toolagent 89%〔算術：d4_nocache 的 compulsory ÷ d4_sim 的 OPT〕。
  - 有快的 SSD 時，最好的延後版已經拿到 oracle 好處的 96.5%–99.99%（CPU 11.6 GiB/s、CPU 2–50%；3.69 那組 gap 本來就是 0）〔算術：(不重用 − 延後)÷(不重用 − OPT)，資料 d4_nocache、d4_gap_posthoc、d4_gap_smallcap〕。
- **為什麼 B 的 SSD 寫入中位數是 0**：CPU 25%、50% 時，「完全不用 SSD」的延後版（LRU＋NONE）就在最好延後版的 5% 以內，所以延後版也可以 0 寫入。差距全部集中在 CPU 10% 那格（見 4.4）。
- **WTP-b（只知道 session 會不會回來）常常比延後版還差**：它不收「不回來的請求」的 block，但其中有 8% 會被別的 session 分叉讀走；而且太粗：只看「新 block 有人讀」的請求，裡面仍有 25.5%（conversation）的新 block 沒人讀〔實測 d4-char，`frac_new_blocks_in_returning_req_but_unused`；這個指標用的是舊的「有人讀」定義，不是 session_returns〕。

### 4.4 差距集中在哪：CPU 小的時候（每格的值）

**SSD＝local、CPU 11.6 GiB/s**（CPU 比 SSD 快的那組）：

| trace | CPU 佔工作集 | 最好延後版 | gap_T | 延後版在 5% 內最少寫 | OPT 寫 | gap_W |
|:--|:--|:--|:--|:--|:--|:--|
| conversation | 2%（事後） | 2HIT＋WT | 1.5% | LRU＋WB 180,216 | 12,786 | 93% |
| conversation | 5%（事後） | ARC＋WB | 1.4% | LRU＋WB 175,860 | 0 | 100% |
| conversation | 10% | ARC＋WB | 0.74% | LRU＋WB 167,727 | 0 | 100% |
| conversation | 25% | LRU＋WB | 0.12% | LRU＋NONE 0 | 0 | 0% |
| toolagent | 2%（事後） | 2HIT＋WT | 1.5% | LRU＋WB 180,564 | 8,236 | 95% |
| toolagent | 10% | ARC＋WB | 0.54% | LRU＋WB2 35,498 | 0 | 100% |
| toolagent | 25% | LRU＋WB | 0.02% | LRU＋NONE 0 | 0 | 0% |

來源：〔實測 20261010-055356-d4-gap、20261010-060105-d4-gap3b〕。寫入單位是 block（64 MiB）。

- 白話：CPU 只有工作集 5–10% 時，「知道誰會被重用」的 oracle **完全不需要 SSD**（因為 76% 沒人要的 block 根本不進 CPU）；conversation 的延後版要嘛寫 10 TiB 到 SSD，要嘛慢超過 5%。
  - 例：conversation、CPU 10%：LRU＋WB 寫 167,727 個 block＝10.2 TiB；第 2 次讀到才寫 SSD 的 LRU＋H2（SGLang selective 式）寫 44,144 個、趕出時讀過 ≥2 次才寫的 LRU＋WB2（KVBM 式）寫 39,564 個，但兩者 TTFT 都比最好延後版慢 6.4%〔實測 d4-sim；算術〕。
  - 10.2 TiB 在 trace 的 58.9 分鐘內寫完，要平均 2.96 GiB/s，**超過** local SSD 的寫入速度 1.91 GiB/s〔算術；tier_params.json〕。在真實時間下延後版寫不完——這是 D1（併發、寫入積壓）的範圍，這裡的模型量不到。
- CPU 3.69 GiB/s 那組：CPU（16.9 ms/block）比 local SSD（8.95 ms）還慢，CPU 層沒有用，所以結論只剩「寫哪些 block 到 SSD」。OPT 只寫會被重用的 44,144 個，延後版要寫 167,727–183,300 個（gap_W 74–78%）；CPU ≥25% 時 LRU＋H2 在 5% 內，gap_W＝0〔實測 d4-gap〕。

### 4.5 敏感度（事先登記，只報告）

| 敏感度 | 結果 | 來源 |
|:--|:--|:--|
| **SSD＝NFS（讀 0.33 GiB/s，大多比重算慢，等於只有 CPU＋重算）** | OPT 的 gap_T：conversation CPU 10% 7.0%／9.7%（3.69／11.6 GiB/s），25% 1.4–2.0%，50% 0.3–0.5%；toolagent 10% 4.3%／6.8%，25% 以上 <0.3%。WTP-a＝OPT。WTP-b 拿到 5.5%／7.4%（conversation 10%） | 〔實測 d4-gap〕 |
| 事後加做 NFS＋CPU 2%、5% | conversation 5%：OPT 12.5%／16.9%，延後版只拿到 oracle 好處的 54%（11.6）；toolagent 5%：9.3%／14.2%；2%：conversation 13.6%／17.5%、toolagent 11.8%／17.0% | 〔實測 20261010-060105-d4-gap3b；算術〕 |
| Cake 重疊模型（前段重算、後段載入並行） | 和逐 block 相加差不多：conversation local 10% 11.6 的 gap_T 0.55%（相加 0.74%）；NFS 10% 11.6 是 9.2%（相加 9.7%） | 〔實測 d4-gap，metric＝ttft_cake_s〕 |
| GPU 層放 2,400 個 block 的 LRU（約 150 GiB）〔算術〕 | 差距幾乎不變：local 每格 gap_T 變化 ≤0.06 個百分點；NFS toolagent 10% 從 6.8% 變 7.4%。GPU 層吃掉 conversation 16,490 次、toolagent 143,227 次重用 | 〔實測 20261010-060103-d4-gap2b、d4-sim2 log〕 |

- **NFS 那組是 TTFT 方面唯一有點肉的地方**：沒有快的 SSD、CPU ≤10% 時，「知道誰會被重用」值 7–17.5% 的總 TTFT。
- 但它的位置正好在第 2 節的「可能」帶（5–15%）上下，而且是**敏感度**，不能改主判定。
- 而且 4.6 顯示這份價值不需要在寫入時用。

### 4.6 事後：同一份知識「延到淘汰時才用」行不行（延後測試）

這不在事先登記裡，是看到結果後才加的對照組，照 10 §2 的「延後版」定義：
- **KEVa**：每個 block 都先收進 CPU（LRU）；要騰位置時，先趕「之後沒人讀」的，沒有才趕最久沒用的；被趕出且之後還有人讀的才寫 SSD。知識和 WTP-a 一樣，只是延到淘汰時才用。
- **KEVb**：同上，但知識換成「session 會不會回來」。

| 格子 | WTP-a（寫入時用） | KEVa（淘汰時用） | 來源 |
|:--|:--|:--|:--|
| local、11.6 GiB/s，全部 CPU 大小（2–50%），兩個真實 trace | gap_T、gap_W 完全相同 | 完全相同 | 〔實測 d4-gap2b、d4-gap3b〕 |
| NFS，全部格子 | 完全相同 | 完全相同 | 同上 |
| local、3.69 GiB/s | gap_T＝0 | gap_T −3.7% 到 −14.8% | 同上 |
| CPU 寫入次數（conversation、10%、11.6） | 44,144 | 182,790（4.1 倍） | 〔實測 d4-sim2〕 |

- **白話**：除了「CPU 比 SSD 慢」那組，同一份知識延到淘汰時才用，TTFT 和 SSD 寫入都一模一樣。
- **3.69 那組的差別不是時機，是規則**：那組最好的做法是「會被重用的 block 直接寫 SSD」（SSD 比 CPU 快）。KEVa 只在淘汰時寫 SSD，所以輸。改成「閒置時把 CPU 裡還有人要的 block 背景寫進 SSD」，寫的集合就和 WTP-a 一樣；在一次一個請求的模型裡，寫的時間不影響 TTFT〔判讀，沒有另外模擬〕。
- **唯一留下來的差別**：延後版會多寫 4 倍到 CPU（GPU→CPU 搬移）。這只在 GPU→CPU 搬移是瓶頸時才有意義，這要看 D1／D2 的結果〔判讀〕。

### 4.7 SCBench（共享 context、一寫多讀）

| 任務 | 例數 | 每個 context 被問幾輪（中位） | context 長度（token，中位） | 2-hit 錯過的重用比例 |
|:--|:--|:--|:--|:--|
| kv、prefix_suffix、repoqa、summary、vt、many_shot | 54–100 | 5 | 2.6 萬–12.5 萬 | 25% |
| mf | 100 | 6 | 9.0 萬（context 是整數陣列，用 JSON 字串計 token） | 20% |
| repoqa_and_kv、summary_with_needles | 70–88 | 8 | 6.8 萬–10.3 萬 | 14% |
| choice_eng、qa_eng、qa_chn | 35–69 | 4–5（qa_chn 2–11 輪） | 14.9 萬–111 萬 | 25–35% |

來源：〔實測 20261010-054738-d4-scbench2〕。token 用 Llama-3.1-8B tokenizer 算，每個任務最多 1,000 例（實際全部都 ≤100 例，所以全算了）。

- **白話**：SCBench 裡**每一個** context 都被問 2–11 輪（中位 4–8 輪），也就是至少被重用 1 次。寫入時「會被重用」這個資訊是 100% 真，等於沒有資訊：全部都收（LRU 寫穿）就對了。
- 2-hit 在這裡反而吃虧：錯過第一次重用（14–35% 的重用）。
- SCBench 沒有定義到達時間和交錯方式，所以**沒有做容量模擬**（NOT_MEASURED）；任何交錯方式都是我自己編的。

## 5. 延後版／更簡單的做法＋前作

**更簡單的做法（本次實測）**：
- 有快的 SSD 時：ARC 或 LRU＋「被趕出 CPU 才寫 SSD」（CPU 2% 時是 2HIT＋寫穿）就拿到 oracle 96.5% 以上的 TTFT 好處〔算術，見 4.3〕。
- CPU ≥25% 工作集時：LRU、完全不用 SSD，就在 oracle 的 2.0% 內（CPU 11.6 GiB/s；conversation 25% 是 2.0%，其餘 ≤0.5%）〔實測 20261010-054330-d4-sim；算術〕。
- 同一份「誰會被重用」的知識延到淘汰時才用（KEVa），結果一樣〔實測，見 4.6〕。

**前作**（子 agent 打開原文查核；我另外抽查了 LPC、Harvard、Alibaba WA、Mooncake、Bidaw、Pensieve 的原文段落，標「抽查」）：

| 前作 | 用什麼資訊決定存不存／趕誰 | 寫入時才有？ | 和本方向的關係 |
|:--|:--|:--|:--|
| **LPC**（Yang, Li, Li, Lloyd；NeurIPS'25） | 用對話文字預測「會不會續聊」，插入時預測、之後隨時間衰減，排 GPU prefix cache 的淘汰順序 | 預測在插入時做，但用的是對話文字，之後也還在 | **最接近 (b)**。和 oracle（完美知道會不會續聊）比過：只補回 LRU→oracle 差距的 22–66%〔原文 p.23，抽查〕；預測器 MCC 0.28–0.36〔原文 p.21，抽查〕 |
| **KVCache Cache in the Wild**（Wang, Han, Wei 等；ATC'25；Alibaba） | 請求類別＋第幾輪 → 重用機率曲線，淘汰時依閒置時間查表 | 類別寫入時就知道，但在淘汰時用 | 真實 trace；WA 比最好的基準只高 1.5–3.9% 命中率〔原文 p.12，抽查〕；沒和 Belady 比 |
| **When Fancy Eviction Fails**（Liu, Yu, Yang；arXiv 2609.28870） | 14 種策略＋Belady，生產 trace | — | 「拿到第 2 次命中後，之後再被重用的機率從 69.4% 升到 88.7%，所以頻率適合當准入、不適合排序」〔原文 p.8，抽查〕。**支持延後版（2-hit 准入）** |
| **Mooncake**（Qin 等；FAST'25；arXiv 2407.00079） | CPU pool 用 LRU；說「不可能準確預測未來使用量」 | — | LRU 比 LFU、LengthAware 好；>50% 的 block 從沒被用〔原文 p.7–8，抽查〕 |
| **CachedAttention**（Gao 等；ATC'24） | 排隊中的請求（look-ahead）決定預取與淘汰 | 不是寫入時，是排程時 | 和 LRU／FIFO 比，沒有 oracle〔原文 p.7、p.11〕 |
| **Pensieve**（Yu, Lin, Li；EuroSys'25） | 重算成本 ÷ 閒置時間，前段先趕 | 全是事後觀察 | 比 LRU 的 CPU 命中率最多高 4.4 個百分點〔原文 p.12，抽查〕 |
| **Marconi**（Pan 等；MLSys'25） | 純輸入 prefix 第 2 次出現才收（第 3 次才命中） | 延後（觀察後） | 就是 2-hit 准入〔原文 p.5–6〕 |
| **Bidaw**（Hu 等；FAST'26） | 上一輪回答長度預測下次重用距離；背景 Belady ghost cache | 回答長度在回合結束時知道 | 換成 Poisson 合成時間戳後就沒效〔原文 p.12，抽查〕 |
| **KVFlow**（Pan 等；arXiv 2507.07400） | agent 步驟圖算「幾步後會執行」，排淘汰與預取 | 應用在請求時給 | 比 SGLang（含 HiCache）快最多 1.83×／2.19×〔原文 p.1〕 |
| **Continuum**（Li 等；arXiv 2511.02230） | 依工具名稱的歷史時長分布設 TTL，把 KV 釘在 GPU | 工具呼叫產生時決定 | 預測「多久回來」，假設一定回來〔原文 p.6–7〕 |
| **SGLang HiCache** | `write_through_selective`：`hit_count ≥ 2` 才寫 host | 延後 | 程式碼：`write_through_threshold = 1 if write_through else 2`〔子 agent 讀原始碼，未由我抽查〕 |
| **Dynamo KVBM** | 開 disk 時，只有 frequency ≥2 的 block 才從 CPU 下放 SSD，預設開，為了 SSD 壽命 | 延後 | 〔官方文件 v0.9.0，子 agent 讀，未由我抽查〕 |
| EfficientAgent、HBF LRU-K、Lachesis | 容量條件准入、第 K 次才寫、依壽命放 | 見舊卡片 | 已在 `research_20261009_explore/cards/` |

- **結論**：
  - 「用預測的未來重用排淘汰順序」已經很多人做（LPC、WA、Bidaw、KVFlow、Continuum、CacheWise〔子 agent 讀，未抽查〕）。
  - 「用觀察到的次數做 2-hit 准入」是業界預設（SGLang、KVBM、Marconi）。
  - 子 agent 沒找到「用寫入時預測決定寫不寫 CPU／SSD，並和延後版正面比較」的論文〔未查證：查不到不等於沒有〕。但本次 4.6 顯示：就算沒人做，這個比較的答案很可能是「延後版一樣好」。

## 6. 如果要繼續

**有看頭的是什麼、不是什麼**：
- 是：「知道誰會被重用 → 沒人要的 block 不進 CPU、不寫 SSD」，在 CPU ≤10% 工作集時能省下幾乎全部 SSD 寫入（或 8.6–12.5% TTFT）。
- 不是：「寫入時決定」。同一份知識在淘汰時用，結果一樣（4.6）。所以這條路如果走，題目是**SSD 准入的重用預測**，對手是 2-hit（KVBM、SGLang selective）和 LPC／WA 這類預測器，不是 S5 的延續。

**建議的小實驗（各一天內）**：
1. **換成真實預測器，不要用 oracle**（最該先做，決定這條路值不值得）。
   - 例如：「這一輪的回答長度」（Bidaw 說和下次重用距離高度相關）、「輪數＋閒置時間」、「block 在請求裡的位置（最後幾個 block 是下一輪最可能讀的）」。
   - 指標：同 TTFT（±5%）下的 SSD 寫入，CPU 2%／5%／10%。和 LRU＋H2（SGLang selective 式）、LRU＋WB2（KVBM 式）、LRU＋NONE 比。
   - 停損：真實預測器在 CPU 5% 時省不到 KVBM 式的 20% SSD 寫入。
   - 注意：LPC 的預測器只補回 22–66% 的 oracle 差距〔LPC 原文 p.23〕，所以先預期只有一部分。
2. **把 D1 的寫入積壓模型接上這份 trace**（唯一可能讓「寫入時」翻盤的地方）。
   - 理由：CPU 10% 時，延後版要在 58.9 分鐘內寫 10.2 TiB 到 SSD，平均 2.96 GiB/s，超過 local SSD 寫入 1.91 GiB/s〔算術〕。寫不完時延後版只能丟，而 oracle 根本不需要寫。
   - 另外延後版會多寫 4 倍到 CPU（4.6）；如果 GPU→CPU 搬移是瓶頸，這會變成 TTFT。
   - 判準照舊：要贏過 KEVa（同知識、淘汰時用）≥5%，才算「寫入時」有效。

**不值得做**：
- 在有快 SSD、CPU ≥25% 的設定上找寫入時資訊的好處（TTFT 差距 <0.3%；不寫 SSD 的 LRU 就在 oracle 2% 內）。
- 用 SCBench 做容量模擬（每個 context 都會被重用，沒有資訊可預測）。
- 「session 會不會回來」這種請求層級的資訊（WTP-b）：在主設定常常比延後版還差（4.3）。

## 7. 失敗與異常

| 時間 | run_id | 問題 | 完整錯誤 | 處理 |
|:--|:--|:--|:--|:--|
| 05:43 | 20261010-054350-d4-scbench | scbench_vt 沒有 `context` 欄 | `File ".../code/m8_trace_oracle.py", line 828, in cmd_scbench: ctx = d["context"] ~^^^^^^^^^^^ KeyError: 'context'` | 查資料：vt 的長 context 放在 `input` 欄（另有 `length` 欄）。程式改成沒有 `context` 時用 `input`，重跑 20261010-054738-d4-scbench2（exit 0）。**`d4_scbench.csv` 裡留有失敗那次寫出的 11 列**（run_id 20261010-054350-d4-scbench，數值和重跑相同）；本文只用 d4-scbench2 的列 |
| 06:00 | 20261010-060056-d4-gap2、20261010-060057-d4-gap3 | `--rule-label` 的值有空白，`runsh` 把參數用 `$*` 寫進 cmd.sh，空白被拆開 | `m8_trace_oracle.py: error: unrecognized arguments: KEVa/KEVb）、SSD=local、GPU=0、ttft_total；6 格中位數（不改判定）`（gap3 同樣錯誤，exit 2） | 改用沒有空白的標籤重跑：20261010-060103-d4-gap2b、20261010-060105-d4-gap3b（exit 0）。這兩次失敗沒有寫出任何 CSV |

**異常與限制**：
- **偏離事先登記（最重要）**：§2.2 的延後版表只列了 7 個策略，但程式在開跑前就放了 25 個組合（多了「完全不寫 SSD」「趕出時讀過 ≥2 次才寫」，以及 2HIT／SLRU／ARC／S3-FIFO 配各種 SSD 規則）。我先用 25 個算出「死路」，寫報告時才發現和 §2 的文字不一致，補算逐字的 7 個（20261010-060621-d4-gap-prereg），結果是「有看頭（gap_W）」。正式判定用逐字版；兩個都在 §4.3。另外 WTP-a 在程式裡多了一個 SSD 規則（WTR：寫入時就把會被重用的 block 寫穿 SSD），§2.2 只寫了「被趕出才寫」；判定時取 WTP-a 兩種規則裡 TTFT 較好的。看成本的 Belady 是抽樣近似（每次抽 32 個），不保證最佳。
- **第一版「session 會回來」的定義失真**：用「請求的任何 block 之後有人讀」，conversation 99.99% 都算回來（因為全部共用 block 0）〔實測 d4-char，`frac_requests_return`〕。主模擬跑之前已改成「之後有請求接在它最後一個完整 block 後面」（`session_returns()`）。舊指標仍留在 `d4_trace_char.csv`，不要引用。
- **CPU 3.69 GiB/s 比 local SSD 讀 6.98 GiB/s 還慢**，這組裡 CPU 層沒有用處。這是沿用第一階段的參數（3.69 是 vLLM 的實測、6.98 是裸讀檔案），不代表真實系統的 SSD 路徑也這麼快。
- **事後加的分析**（KEVa／KEVb、CPU 2%／5%、nocache 基準）都在看到主結果後才加。它們不改判定，只用來解釋：KEVa 拿掉了「寫入時」的功勞；CPU 2%／5% 把資訊有價值的區域畫得更清楚（在那裡差距更大）。請照這個前提讀。
- f(i) 在 i ≥ 80（40K token 以後）是線性外插；conversation 重用存取的位置 p90 是 73、p99 是 167〔實測 d4-char〕，所以大部分在量測範圍內。
- 一次服務一個請求、沒有排隊、SSD 容量無限、寫入不花時間——所以延後版的寫入永遠不會被卡住（這正是第一階段的最大限制，見 FULL_REPORT §7）。
- 子 agent 讀的前作，我只抽查了 6 篇的關鍵段落；其餘標〔原文 p.X〕的頁碼是子 agent 讀到的，引用前要再核對。
