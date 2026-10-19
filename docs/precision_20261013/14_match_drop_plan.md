# 14　讓「存」和「算」匹配（M1、M2）＋直接丟掉不重要的 block（D）（事先登記；只報告）

> **為什麼做**（使用者 10/19）：看了 Cake 論文，「存（載入）和算差太多，Cake 就沒意義」→ 我們的實驗差在哪？想讓兩條線的速度匹配；另外，論文說不重要的 token 可以直接丟，連載入都不用。使用者回覆「全部都做」。
> **起點**：11 問題三 §3c（`code/m10_cake_ratio.py`，10/19 01:4xZ）。「只載 ÷ 只算」的比在真實 CPU 層是 0.007–0.022，這時 Cake 只有單一做法的 0.50–0.86 倍；在 b48 是 0.40–2.6，這時是 1.51–2.04 倍。
> **寫的時候**：44a（長 context 速度）在跑。M1、M2、D 都**還沒有任何結果**。只用過已經 commit 的主線格子、42b 的方法海選、43 複賽的品質。
> **地位**：探索，只報告。主線（07）、大掃描（09）、13 號的判定都不改。

## 1. 要回答的問題

- **M1（還原單位）**：把一塊從 512 token 切成 256、128，兩條線就能在更細的位置相遇（少一點「一條線做完在等另一條」）。這在慢的層有沒有幫助？
  - 42b 已經在真實 CPU 層量過（11 問題一 §3b）：切細會變慢，原因是每塊有固定開銷。這次補慢的層。
- **M2（用格式去匹配）**：壓縮會把「只載 ÷ 只算」往下移。在慢的層（b16、b32、b48、NFS 0.331 GiB/s），比 INT4 更狠的格式（K4V2、INT2、MXFP4）還會不會繼續變快？
  - 什麼時候「比接近 1」？
  - 每一層最快、而且品質過得了 43 複賽 ε 的格式是哪個？
- **D（直接丟掉）**：還原時直接不要某些 block（不載、不算，attention 也看不到）。這樣品質掉多少？省多少時間？
  - 「誰來決定丟哪些」是重點：
    - **寫入時決定**：prefill 結束時看 attention（ATT），或照位置（F／M／B）。
    - **還原時決定**：這時已經知道下一個問題（QRY）。
  - 兩輪的題目可以看出多輪的問題：照第 1 輪問題丟的，第 2 輪會不會出事（SCBench，ICLR 2025）。

## 2. M1：還原單位（速度）

- 程式：主線 `m10_speed.py grid`，加 `M10_UNIT`（已經有，42b 的 G1、G2 用過）。拆的方法：S0（K\*，融合 kernel；`M10_DEQUANT_FAST=1`），會合：claim（Cake）。
- 格子：
  - 兩個模型；單位 512／256／128；層：真實 CPU、b16、b32、b48（每個模型自己的 `bw.json`）。
  - 配置：E0、E1s（INT4 KIVI）；GPU 閒／忙（busy2）。
  - 2 launch × 5 rep，launch 從 600 起。
- 報：每格 E1s、E0 的還原時間中位數、會合點；128、256 對 512 的比。
- 事先寫的預期（還沒有任何慢層的數字）：
  - 真實 CPU 層：和 42b 一樣變慢。
  - b32、b48：最多快「半塊的時間」的量級（≤ 5%）。每塊的固定開銷可能把它吃掉。

## 3. M2：用格式去匹配（速度）

- 程式：同上；新格式用 `M10_CODEC_X=1 M10_DEQUANT_FAST_X=1`（13 號 Q3 的融合 kernel，43d 在 GPU 上逐位元過）；S0；claim。
- 層：b16、b32、b48（每個模型的 `bw.json`）＋ **nfs 0.331 GiB/s**（12 號用的 NFS 級，`m10_tier2.slow_tiers()`；寫進 `results/m10_match_mi300x/bw_<m>.json`）。
- 格式（每個一個 grid）：主線 FP8（`fmt_of(m, fp8)`）、int4_kivi、kv-int4_kivi-int2_kivi、int2_kivi、mxfp4。
- 每個 grid：
  - 配置 E1s（第一個 grid 另加 E0）。
  - 參考線：只算、只載 E1s（第一個 grid 另加只載 E0）。
  - GPU 閒／忙；2 launch × 5 rep，launch 從 700 起。
- 報：
  - 每格「只載 ÷ 只算」、Cake 的時間、Cake 對單一做法的增益（和 11 問題三 §3c 同一個算法，`m10_cake_ratio.py` 的延伸）。
  - **每一層最快的格式**，以及「43 複賽 E1s 的 ε 過了的格式裡最快的」（Llama：只有 INT4 KIVI 過；Qwen：INT4 KIVI、K8V4 過；K4V2 兩個模型都沒過）。
- 事先寫的預期：
  - 位元越少，絕對時間越短，一直到 NFS；但在 b16 可能被拆的時間抵消。
  - 比最接近 1 的格式不一定最快（11 §3c：b32 換 INT4 後比離 1 更遠，但絕對時間更短）。

## 4. D：直接丟掉 block（品質＋速度）

### 4.1 怎麼丟（定義）

- 丟掉＝那些 block 的 KV 不存、還原時不載也不算，之後的 attention（問題、生成）完全看不到那些位置。
- **實作**：`KVModel.key_keep`（每個位置一個 bool）。設了之後 attention 用「causal 且 key_keep」的遮罩。
  - 這和「把那些 key／value 從 cache 拿掉」數學上相同：softmax 只在留下的位置上做。
  - 其他位置的 RoPE 位置不變，和 H2O、SnapKV 一樣。
  - 沒設的時候，路徑和原本逐字相同。
- block 0 一律保留（attention sink）。k＝round(佔比×63) 個 block 從 1–63 裡丟。
- 遮罩路徑和原本的 SDPA 路徑數值上不一定逐位元相同，所以加一個**對照配置 MASK0**（遮罩路徑、什麼都不丟）。它和 E0 的差就是「換路徑本身的雜訊」。
- 只看讀法 A（全部載入、不重算）：真實 CPU 層 G5b 選的就是這個（13 §3b）。
  - 慢的層 Cake 會重算前面的 block，丟在前面的會被算回來，品質只會比讀法 A 好。所以讀法 A 是下限。

### 4.2 丟哪些（放法）

- F／M／B：最舊／中間／最新的 k 個（和 09 同一個定義）。
- R0–R4：隨機 5 個 seed（09 同一個定義）。
- **ATT**：prefill 結束時，最後一個 block 對每個 block 的 attention 量最少的先丟。和 09 同一段程式，SnapKV 式的「寫入時觀察窗」；最後一個 block 永遠留著。
- **QRY**（新）：還原時已經知道第 1 輪的問題：用問題的每個 token 對每個 block 的 attention 量（所有層、head 加總，BF16 的 KV）最少的先丟。
  - 這是「讀的時候才決定」（Quest、InfiniGen 式）的上限：用的是完整的 attention，實際系統只能近似。
- ORC：作弊上限：第 1 輪答案所在的 block 最後才丟（09 同一個定義）。

### 4.3 D1 海選（teacher forcing；GPU）

- 程式：`m10_sweep.py screen`，格式 `drop`（新；bits＝0、存的位元組＝0）。
- 題目：每個模型 T2–T5 各 24 題，seed 從 60000 起（不和 10000／20000／40000／50000 重疊），32K。
- 放法：F M B R0–R4 ATT QRY ORC；佔比 10、25、50、75%；加 ALL、E0、MASK0。
- 量：
  - 對 E0 的 KL、標準答案 log 機率、first_ok（09 §3 同一套）。
  - 第 1 輪的問題（screen 只用第 1 輪；ORC、QRY 都看第 1 輪）。
- 報（`code/m10_drop.py screen-report`）：
  - 每個（放法, 佔比）：first_ok 對 E0 的差、KL 中位數、對隨機（R0–R4 平均）的配對差＋題目為單位的 bootstrap 95% CI。
  - 沒有多重比較校正，只用來挑 D2，和 09 §4.2 一樣。

### 4.4 D2 複賽（生成；GPU）

- 程式：`m10_quality.py run`，配置 `SW-<放法>-<佔比>@drop`（主線同一套出題、生成長度、計分、兩輪）；讀法 A。
- 題目：新 seed（70000 起），T2–T4 各 45 題、T5 15 題。
- **配置（寫死，不看 D1 的結果）**：
  - 佔比 25%、50%。
  - 每個佔比：R0、R1、R2、F、B、ATT、QRY、ORC。
  - 加 E0、MASK0。
  - 共 18 個。
  - D1 只拿來看形狀（例如 75% 是不是全部壞掉），不改 D2 的配置。
- 判準（只報告；用 07 的 ε 規則，`m10_analyze.eps_check`）：
  1. 每個配置對 E0 的 ε（CI 下界 > −2，而且每個任務族、每一輪 ≥ −5）。
  2. ATT、QRY、F、B 對隨機（R0–R2 平均）的配對差，bootstrap 95% CI。同一個佔比的 4 個比較用 Holm（α＝0.025 單邊）。
  3. **多輪**：同一個配置，第 2 輪對 E0 的差減第 1 輪對 E0 的差（QRY、ORC 照第 1 輪的問題選，預期第 2 輪掉更多）。
  4. MASK0 對 E0 的差和 ε 一起報（如果 MASK0 自己就沒過 ε，D 的品質結論只能寫「下限」）。

### 4.5 D3 速度（GPU）

- 程式：`m10_speed.py grid`；新配置 `<底>-D<放法><佔比>`。底＝E0（BF16）或 E1s（INT4 KIVI）；放法 F、B、R（R 用和 R25 同樣的 seed）；佔比 25、50。
- 還原器 `Restorer.drop`：載入線跳過被丟的 block。重算線照 Cake 從前面做；前面被丟的 block 還是會被算回來（那段需要完整的前綴）。KV 檢查只看實際載入／重算的 block。
- 層：
  - 真實 CPU 層：全部載入（`--extra-modes loadall`）和 Cake。
  - b16、b32、b48：Cake。
  - nfs：Cake，只跑 BF16 底。
- GPU 閒／忙；2 launch × 5 rep，launch 從 800 起；S0（K\*）。
- 報：每格的還原時間中位數，和「同一個底、不丟」的比。
- 限制：新一輪的 attention 仍然跑在整段長度上（遮罩），所以 TTFT 偏保守。只把還原時間當主要數字。
- 事先寫的預期：
  - 慢的層丟 B（最新的）省最多：載入線從後面開始。
  - 丟 F（最舊的）在慢的層省很少：那些本來就會被重算。
  - 真實 CPU 層全部載入，省的時間大約和丟掉的位元組成比例。

### 4.6 事先寫的預期（品質）

- ATT（寫入時）和隨機差不多。09 海選和 43 複賽裡，寫入時的 attention 挑法沒有贏隨機。
- QRY 比隨機好很多，第 1 輪接近 ORC；但第 2 輪掉得比第 1 輪多（多輪問題；SCBench）。
- 丟 25% 在找答案的任務（T2、T5）就會掉很多；ε 大概只有 ORC、QRY（第 1 輪）的低佔比過得了。

## 5. 順序、時間、上限

- 佇列（使用者 10/19：「全部都做」；放在 44d 後面、只報告的工作 45 前面）：
  - `44e0_md_smoke`：新程式在 GPU 上的冒煙。
    - 遮罩對拿掉 key 的數值差。
    - MASK0 對 E0。
    - 丟掉真的會改變輸出。
    - 速度路徑的 drop 配置、KV 檢查、nfs 層。
  - → `44e1_d1_screen` → `44e2_d2_confirm` → `44e3_d3_speed` → `44e4_m2_fmt` → `44e5_m1_unit` → `44ez_md_report`（CPU）。
- 〔估計〕GPU：冒煙 0.3、D1 1、D2 1.5–2、D3 0.5、M2 2、M1 1 小時，合計約 6.5–7 小時。
- 上限 9 小時。快超過時照這個順序砍：先砍 M1 的 b16，再砍 M2 的 mxfp4，再砍 D2 的 50%。
- 冒煙沒過：用到那段程式的 stage 寫 NEEDS_HUMAN、跳過，其他照跑。
- 失敗處理同 RUNLOG 14.5：寫 NEEDS_HUMAN、跳過那段的其餘部分、exit 0。
- 計時的 stage（D3、M1、M2）跑的時候不插任何東西。gpu_guard 不是 CLEAN 的格子要重量。

## 6. 不做的事

- 不做「丟掉後壓縮 cache、讓 attention 真的變短」的速度：KVModel 的緩衝是固定長度。新一輪的 attention 時間不當結論。
- 不做讀法 B 的 D 品質：讀法 A 是下限，上面寫了。
- 不調 QRY／ATT 的超參數（觀察窗大小、加總方式），照寫好的定義跑一次。
- 不改主線、09、13 的任何判定。

## 7. 程式（寫在登記之後）

- `code/m7_model.py`：`KVModel.key_keep`（預設 None＝原本的路徑）。
- `code/m10_drop.py`：放法（含 QRY）、問題的 attention 量、遮罩的設定與清除、D1 的報告、D2 的判讀、M1／M2／D3 的表。
- `code/m10_sweep.py`：格式 `drop`、放法 QRY、MASK0（預設的海選不會走到）。
- `code/m10_quality.py`：`SW-*@drop`（預設不會走到）。
- `code/m7_restore_harness.py`：`Restorer.drop`（預設空集合＝原本的路徑）。
- `code/m10_speed.py`：`<底>-D<放法><佔比>` 配置。
- CPU 測試：`code/m10_queue_mi300x/tests/match_drop_cpu.py`。

## 8. 補充（10/19 01:5xZ）：D 改成大規模丟 token 掃描（DS）

**這時候還沒有任何 M1、M2、D 的結果，GPU 上也還沒跑任何這一號的東西。**

> **使用者 10/19**：「改狀態（精度）感覺差距不大，短 context BF16、FP8 都差不多 → 可以從直接刪掉 token 做大規模掃描，跟看各種 paper。」
> **為什麼合理**：32K 時好的低精度格式本身就不傷答對率（主線 K1／K3、INT4 全過、43 複賽 0／12），所以「放哪裡」沒有東西可救。丟掉 token 會真的傷品質，挑法之間才會拉開。

### 8.1 改了什麼

- §4.3 的 D1、§4.4 的 D2 換成下面的 DS1、DS2。§4.1 的定義（遮罩＝拿掉，MASK0 對照）、§4.5 的 D3 速度、M1、M2 都不變。
- 挑法改成照文獻。分數的定義照 NVIDIA kvpress 0.5.5（Apache-2.0）的寫法**重寫**（不是直接用它的程式：它掛在 HF 的 cache 上，我們的 KV 緩衝是自己管的）。原始碼只用來讀，放在 scratch，沒有執行。
- 文獻與每個挑法怎麼對應：`docs/precision_20261013/cards/P08_token_drop_literature.md`。

### 8.2 挑法（分數高的留下；丟掉分數最低的）

| 代號 | 文獻 | 什麼時候算 | 分數 |
|:--|:--|:--|:--|
| STREAM | StreamingLLM（ICLR'24） | 寫入時 | 留前 4 個 token（sink）和最近的；丟最舊的 |
| NEW | 對照 | 寫入時 | 丟最新的（和 STREAM 相反） |
| RND0–RND2 | 對照 | 寫入時 | 隨機（3 個 seed） |
| SNAP | SnapKV（NeurIPS'24） | 寫入時 | context 最後 64 個 token 當觀察窗，對前面每個 token 的 attention 平均，avg-pool 5，GQA 組內平均；觀察窗自己保留 |
| TOVA | TOVA（2024） | 寫入時 | context 最後一個 token 的 attention，所有 head 平均 |
| H2O | H2O（NeurIPS'23）／ObservedAttention | 寫入時 | 累積 attention ÷ 看得到這個 token 的 query 數。**近似**：只用均勻分布的 8 個 64-token 觀察窗（block 7、15、…、63 的最後 64 個），不是全部 32K 個 query（全部算一題要幾分鐘） |
| PYR | PyramidKV（2024） | 寫入時 | SNAP 的分數，每層的預算照 pyramid（window 64、beta 20），只做 per-head |
| ADA | AdaKV（2024） | 寫入時 | SNAP 的分數，同一層的 head 之間一起挑，每個 head 至少留預算的 20%，只做 per-head |
| EXPATT | Expected Attention（kvpress，2025） | 寫入時 | 用 context 全部 query（去掉前 4 個）的平均、共變異數，加上接下來 512 個位置的平均 RoPE，算預期 attention × ‖V‖ |
| KNORM | L2-norm（Devoto et al. 2024） | 寫入時 | −‖K‖（norm 小的留下） |
| KEYDIFF | KeyDiff（2025） | 寫入時 | −cos(K, 平均 K)（越不像平均的留下） |
| QRY | SnapKV 用真的問題當觀察窗（＝單輪 SnapKV、FINCH 式） | **還原時**（知道第 1 輪問題） | 第 1 輪問題每個 token 對 context 每個 token 的 attention 平均，avg-pool 5，GQA 組內平均 |
| ORC | 作弊上限 | — | 第 1 輪答案所在的 block 全部保留，其餘照 RND0 |

- 不做的（照實寫）：
  - KVzip（要把整段 context 重建一次並記下 attention，一題幾分鐘）。
  - DuoAttention、Q-Filters、KVP、ForesightKV、LookaheadKV（要訓練或要每個模型的權重；CLAUDE.md §8）。
  - Quest（逐步選，不是丟；QRY 是「知道問題」的上限）。
  - 需要 eager attention 的完整 H2O。

### 8.3 粒度

- **T（per-head token）**：每個（層, KV head）各自留 n_kept＝max(1, int(長度×(1−佔比))) 個 token（kvpress 的 ScorerPress）。PYR、ADA 照各自的預算。這是文獻原本的做法。
- **P（page，16 token）**：所有層、head 一起丟同一組 page。page 分數＝每個（層, head）上「page 內 token 分數平均」的百分位排名，再對所有（層, head）平均。第一個 page 一律保留。
- **B（block，512 token）**：同 P 的算法，但以 block 為單位，block 0 一律保留。**只有這個粒度能直接換成還原的速度（D3）**。
- 位置類（STREAM、NEW、RND）在 P、B 照位置直接挑；ORC 只做 B、P。
- 被丟的比例 10、25、50、75、90%。

### 8.4 DS1 海選（teacher forcing，第 1 輪）

- 程式：`code/m10_drop.py screen`。和 09 海選同一套 Item（prefill 一次、問題＋標準答案一次餵進去），量對 E0 的 KL、標準答案 log 機率（gold_lp）、first_ok、top1，另外記「答案所在 token 留下的比例」。
- 題目：每個模型 T2–T5 各 24 題，seed 從 60000 起；32K。
- 對照：E0、MASK0（全域遮罩全留）、MASK0H（per-head 遮罩全留）。
- 報：
  - 每個（粒度, 挑法, 佔比）的平均差和題目為單位的 bootstrap 95% CI。
  - 對 RND 平均的配對差。
  - 每個佔比的排名。
  - 沒有多重比較校正，只用來挑 DS2。

### 8.5 DS2 複賽（生成，兩輪；挑法寫死）

- 程式：`m10_quality.py run`，配置 `DS-<粒度>-<挑法>-<佔比>`；讀法 A；新 seed 70000 起，T2–T4 各 45、T5 15。
- **選手規則（寫死）**：每個模型、每個粒度（T、B），在 25% 和 50% 兩個佔比上，以 DS1 的 gold_lp 對 E0 的平均差（兩個佔比平均）排名。
  - 取「寫入時」挑法（不含 RND、ORC、QRY、STREAM、NEW）的前 2 名。
  - 再加 STREAM、RND0、QRY、ORC（ORC 只有 B）。
  - 兩個佔比都跑。加 E0、MASK0、MASK0H。
  - 每個模型：T 粒度 5 個挑法 × 2 個佔比＝10，B 粒度 6 個 × 2＝12，加 3 個對照，共 25 個配置。
- 判準（只報告）：
  1. 對 E0 的 ε（07 的 `eps_check`）。
  2. 前 2 名、QRY、STREAM 對 RND0 的差，同一個（粒度, 佔比）裡 4 個比較用 Holm。
  3. **多輪**：第 2 輪對 E0 的差減第 1 輪對 E0 的差，看 QRY（照第 1 輪問題挑）在第 2 輪是不是掉更多（SCBench 說的問題）。
  4. MASK0／MASK0H 對 E0（換路徑的雜訊）。
- 原本 §4.4 寫死的 D2 配置（B 粒度 R0–R2、F、B、ATT、QRY、ORC × 25／50%）被這一條取代：B 粒度的 STREAM＝F、NEW＝B、RND＝R，SNAP 取代 ATT。

### 8.6 時間與上限

- 〔估計〕DS1：每題約 240 個 teacher forcing＋分數約 15 秒，約 2 分鐘／題 × 192 題，4 個 process 一起跑約 2–3 小時。DS2：約 1.5–2 小時。
- 這一號合計約 8–9 小時，上限改成 11 小時。
- 砍的順序：先砍 M1 的 b16，再砍 M2 的 mxfp4，再砍 DS1 的 90%，再砍 DS2 的 P 粒度（DS2 本來就只有 T、B）。
- 佇列：`44e0_md_smoke` → `44e1_ds1_screen` → `44e1z_ds_pick`（CPU，照 §8.5 寫出 DS2 的 jobs）→ `44e2_ds2_confirm` → `44e3_d3_speed` → `44e4_m2_fmt` → `44e5_m1_unit` → `44ez_md_report`。
