# C1b 怎麼跑：選擇性還原（回來的問題只需要少數 block，只搬需要的部分）（第三階段；10/21 寫）

> **狀態**：`code/m16_c1b_restore.py` 的 GPU 路徑**還沒有執行過任何一行**。寫的時候第三階段自己的 job 正在 GPU 上量時間，不能開 GPU 行程、不能 import torch。
> 做過的：`py_compile`；CPU 的 `--self-test`（46 項全過，不 import torch）；靜態檢查（函式裡 import 的名稱都存在、用到的 `KVModel` 屬性都存在、沒有未定義的全域名稱）。
> GPU 路徑真正的驗證是 38da／38db 開頭的 `selfcheck`（§8）。
> 本頁的時間全部是〔估〕，依據寫在 §5；還原時間是〔計算〕，常數的出處寫在 §2.7。沒量到的一律 NOT_MEASURED。
> 設計卡：`experiment_matrix.md` §3「C1b　選擇性還原：不載入 ≠ 丟掉」。閘門 C1a 已過（`experiment_log.md` 的 C1a 節）。

---

## 0. 一句話

使用者帶著第 2 輪問題回來時，不把 64 個 block 全部搬回 GPU，只搬挑中的 k%（k＝10、25、50），其餘留在下層。比兩件事：
- 答對率掉多少（對完整還原，同一題配對）。
- 「留在下層、需要時補載」(b) 有沒有比「直接丟掉」(a) 好。

---

## 1. 檔案

| 檔案 | 內容 |
|:--|:--|
| `code/m16_c1b_restore.py` | 子命令 `selfcheck`、`qtime`、`run`、`report`、`pool-claim`／`pool-status`／`pool-done`；`--self-test` |
| `code/m16_queue/jobs/38da_c1b.sh` | GPU 第 1 段：selfcheck＋qtime＋工作池（預算 105 分鐘） |
| `code/m16_queue/jobs/38db_c1b.sh` | GPU 第 2 段：接著做沒做完的題；都做完了就不佔 GPU |
| `code/m16_queue/jobs/38dz_c1b_report.sh` | CPU：report＋題數與逐題檢查的核對 |
| `code/m16_queue/jobs/c1b_common.sh` | 38da／38db 共用的本體。**不是 job，不要放進佇列** |

輸出：
- `results/m16_stage3_mi300x/c1_selective_restore/`（進 git）：`c1b_summary.json`、`tables_c1b.md`、`c1b_cells.csv`、`qtime_<模型>.csv`。
- `/mlsteam/data/tiara/stage3/c1b/`（不進 git，> 1 MB）：逐列 `c1b_<模型>_s<i>.csv.gz`、`*.checks.jsonl`、`*.gpu_guard.json`、工作池目錄 `pool_<job>_*`。
  - report 的 JSON 記每個原始檔的大小與 sha256。
- 每一列都有 `run_id`、`ts`、`shard`、`n_rows_item`、`own_gpu_intruders`、`host_level`。

既有檔案一個都沒改，只用 import 重用：
- `m16_c1a_predict`：`compute_item`、`block_minmax`、`quest_scores`、`agg_blocks`、`keep_order_from_blocks`，以及 CSV／監控／統計的小工具。
- `m16_s2_equalbytes.keep_tensor`。
- `m10_quality`：`Answerer`、`gen_len`、`run_turns`。
- `m10_tasks`：`make_item`、`score`、`qa_f1`。

唯一照抄的是 `KVModel.forward_span`，抄成 `fwd`。逐行相同，只多一件事：留下每層的 post-RoPE q，補載的 Quest 打分要用。
selfcheck 會在三種遮罩下逐位元比對它和 `KVModel`。

---

## 2. 設計（寫死）

### 2.1 題目

- 和 C1a 同一組：DS2 的兩輪題，seed 70000 起；T2–T4 各 45 題，T5 15 題；32K；參數用 `CANDIDATE_PARAMS.json`。
- 只換處理順序：
  - k 由小到大，每個 k 的三個任務輪替起點。這樣任何一段前綴都涵蓋三個任務、三種位置帶。
  - T5 放最後，時間不夠時先犧牲它。
- self-test 核對過題目集合和 C1a 的 CSV 完全相同（兩個模型各 150 題）。

### 2.2 每題的流程

1. **打分**：`compute_item`，和 C1a 同一段程式。寫入時分數只算本程式要用的那一種。
   每個選法的保留順序，逐題和 C1a CSV 的 `order` 欄逐字比對，結果記在 `order_eq_c1a`。
2. **暫停前的狀態**（完整 KV、sdpa 路徑）：
   - 第 1 輪問題＋greedy 生成，計分記在 `score_t1_full`。
   - 再把歷史（`history_ids`＝第 1 輪標準答案＋換手 token）也用完整 KV 接上。
   - 到這裡的 KV 就是卸載前的快取。context 之後這一小段（tail，約幾十到幾百 token）還原時一律載入。
   - T5 只有一輪：暫停點＝context 剛 prefill 完，「回來」的就是它唯一的問題。
3. **回來**：每個格子各做一次。
   - 設好 `key_keep`（沒載入的 block 遮掉；RoPE 位置不變）→ 接第 2 輪問題 → greedy 生成。
   - 計分用 `m10_tasks.score`，和第二階段同一個計分。生成長度：T5 120、其他 64。
   - 每格只寫第 2 輪問題之後的位置，所以格子之間互不影響。每題結束時核對暫停前的 KV 逐位元沒變，變了就停（規則 2）。

### 2.3 選法

保留順序都是 1..63 的排列，前 m 個載入；block 0 一律載入。

| 選法 | 定義 | 能不能部署 |
|:--|:--|:--|
| QUEST_MEAN | C1a 原樣。第 2 輪問題直接接在 context 後面，**看完整 context** 算出每層的 post-RoPE q，打每個 block 的 K min／max 摘要，略過前 2 層，對（層, q head）平均 | **偏樂觀**：深層的 q 要看過完整 context 才算得出來，還原前拿不到。設計卡指定的選法，設計卡的判定照它 |
| QUEST_BLIND | **新增**。q 用「只看得到 block 0＋tail」的 forward 算，第 2 輪問題放在它真正的位置；摘要與彙總和 QUEST_MEAN 相同 | 可以。只用還原當下 GPU 上真的有的東西 |
| FULLK_MEAN | C1a 的上界：完整 K 的真實 attention 機率 | 不能（oracle）。位元組不含選法本身的成本 |
| WRITE | 寫入時最好的一種：Llama＝KNORM、Qwen＝KEYDIFF。依據是 C1a T2–T4 的 recall@16／@32（+23.6／+26.4；+19.2／+19.3 個百分點） | 可以（卸載時就算好） |
| QRY1 | 第 1 輪問題的 attention（C1a）。T5 只有一輪，所以就是當下的問題（kind＝`qry_target`） | 可以（第 1 輪時就算好） |
| POS | 位置：只載前 k%，也就是 block 1..m（＝C1a 的 NEW 順序） | 可以 |
| RND | 隨機（C1a 的 RND0，每題固定種子） | 可以 |

基準有兩個：
- **FULL**：64 個 block 全載入，走和各格相同的遮罩路徑。判定都對它比。
- **E0_SDPA**：`key_keep`＝None，就是第二階段的 sdpa 路徑。FULL − E0_SDPA＝換路徑的雜訊，另外報。

### 2.4 k 與變體

- **k** ∈ {10, 25, 50}%，佔候選的 63 個 block → m＝6、16、32（和 C1a 的 k＝16、32 對齊）。初始載入 1＋m 個 block。
- **(a) 不補載**＝丟掉。
- **(b) 補載**：
  - 第 2 輪問題 prefill 完先檢查一次，用問題自己的 q；之後每生成 `REFILL_EVERY`＝4 個 token，用這 4 個 token 的 q 再檢查一次。
  - 每次檢查用當下的 q 重打 64 個 block 的 Quest 分數，彙總和 QUEST_MEAN 相同。
  - **門檻規則（寫死）**：當下分數的前 m 名（只在 1..63 裡排）裡還沒載入的 block 全部補載。m＝初始載入的候選 block 數。載入的不再丟回去。
  - 補進來的 block 從下一個 forward 起看得到；已經生成的 token 不重算（和真的系統一樣）。
- (a) 7 種選法 × 3 個 k；(b) 6 種選法（FULLK 是 oracle，不做）× 3 個 k；加上 FULL、E0_SDPA → **每題 41 格**。

### 2.5 位元組（`bytes_row`，照實算）

- **初始**＝(1＋m) 個 BF16 block＋tail。
- **摘要**：用到 Quest 分數的格子（QUEST_*、所有 (b)）才算，＝64 個 block × 2×L×H×D 個 BF16（K 的 min、max）。
  - Llama 8 MiB、Qwen 3.5 MiB，約完整還原的 0.2%。
  - 設計上它常駐 GPU，這裡保守地也算進搬運。
- **補載**＝補載的 block 數 × block 位元組。
- 比例的分母＝完整還原（64 個 block＋tail）。另記不含摘要的比例。
- 例：k＝25%、tail 100 token、含摘要 → 0.2698（兩個模型一樣）。

### 2.6 指標（每題 × 每格一列）

| 指標 | 說明 |
|:--|:--|
| 答對率 | `score`；T4 另記 `f1` |
| 配對差 | 對 FULL、對 E0_SDPA；以題目為單位 bootstrap 10,000 次，95% CI |
| 位元組 | 搬運位元組與比例 |
| 補載 | 次數、block 數、每次補了哪些（`refill_log`） |
| 答案 block 的覆蓋 | `needle_cov_*`、`needle_all_*`，初始與最終各一。答案 block＝m10_tasks 組裝時記的 needle 位置 // 512，C1a 用兩條獨立路徑驗算過。T5 的「答案 block」是常見詞每一次出現所在的 block，只當參考 |
| 同字 | `same_as_full`：生成的 token 和 FULL 完全相同 |
| 時間 | `t_gen_s`：同一張卡有其他行程時作廢 |

### 2.7 還原時間（report 計算；首 token 前要搬的部分）

**公式**
- 只搬 k%＝(1＋m)×block 位元組 ÷ 頻寬。
- 全部搬＝64×block 位元組 ÷ 頻寬。
- Cake＝min_b max(Σ_{i<b} f_i, (64−b)×block 位元組 ÷ 頻寬)，f_i 是每個 chunk 的實測重算時間。

**常數**（都從檔案讀，記 run_id）

| 常數 | 值 | 出處 |
|:--|:--|:--|
| CPU 層 pinned H2D | Llama 49.858／Qwen 48.644 GiB/s | `speed_<m>/bw.json`（`20261015-154558-m10-calibf-llama`、`-154647-…-qwen`） |
| f_i | 64 個 chunk 的重算時間 | 同上 |
| b32 | Llama 1.467／Qwen 0.763 GiB/s | 同上 |
| SSD、NFS 限速參數 | 6.98、0.331 GiB/s | `tier_params.json`（`20261008-130257-m7-c0`、`-130438-m7-c2`） |
| vLLM 版面每片成本 | py_loop＝逐塊 10.17 µs、coalesce 10.16、stage＝CPU 端合併 1.50、zerocopy＝GPU 直接讀 0.575 µs | I4 `copybench_sdma1.csv`（scatter、閒，`20261020-143218-m16-i4-sdma1`） |

vLLM 版面的切法：16-token 的頁、每層 K／V 各一片，一個 512-token block＝32×L×2 片。I4 量的是 Llama 的 32 KiB 片；Qwen 的片是 16 KiB，沿用同一個每片 µs，**是推算**。

**核對**：Cake 公式算出來 Llama NFS 2.599 s、Qwen 1.734 s，和 L5c §4 的表相同。

**k＝25% 的例子**〔計算〕

| 層級 | Llama 只搬 k% | 對全部搬 | 對 Cake |
|:--|--:|--:|--:|
| CPU 層 | 21.3 ms（全部 80.2、Cake 77.7） | 0.27 | 0.27 |
| b32 | 0.72 s | 0.27 | 0.53 |
| NFS 級 | 3.21 s | 0.27 | **1.24** |

- **只搬 k% 只能「載入」**：選中的 block 不連續，不能用前段重算。所以在 NFS 這種慢層級，k＝25% 反而比 Cake 慢。「前段重算＋後段只載選中的」這種混合，品質沒有量，沒有算進來（§9）。
- vLLM 版面、逐塊 copy：k＝25% 要 354 ms，全部要 1,333 ms；zerocopy 是 20 ms／75 ms。
- **Quest 的額外時間**由 qtime 在閒的 GPU 上實測：可部署版 q 的 forward＋選 block，以及每次補載檢查。
- 補載的 block 在 decode 期間才搬，另外列。
- 第 2 輪問題的 prefill 不算：每格相同；而且本 harness 的遮罩路徑仍對全部 key 計算，量不出少算的好處。
- 所以設計卡的「首 token 時間」在這裡＝還原時間（計算）＋Quest 額外時間（實測），**不是端到端實測**。

---

## 3. 事先寫死的成功／停止條件（`judge`，看到數字前寫死）

**範圍**：每個模型、T2–T4，至少 48 題齊全（設計卡：主格 ≥ 48 題）。T5 是全域任務，答案散在整段 context，C1a 也沒有它的 recall；另報、不進判定。

**記號**

| 記號 | 意思 |
|:--|:--|
| ε | 2 個百分點，沿用第二階段 |
| 在 ε 內 | 對 FULL 的配對差 95% CI 下界 ≥ −ε（和 X2 M1 同一個寫法） |
| 搬運 ≤ 30% | 搬運比（初始＋tail＋摘要＋補載）的題目平均 ≤ 0.30 |

query-aware 的兩種選法 M（QUEST_MEAN＝設計卡指定；QUEST_BLIND＝可部署版）**各判一次**：

| 成分 | 定義 |
|:--|:--|
| A | (a) k＝25%：在 ε 內，而且搬運 ≤ 30% |
| B | (b) k＝25%：在 ε 內，而且搬運 ≤ 30%（含補載） |
| R | k＝25% 的 (b) − (a) 配對差 CI 下界 > 0，也就是「(b) 明顯好過 (a)」 |
| STOP | k＝50% 時，QUEST_MEAN、QUEST_BLIND × (a)、(b) 四格的配對差**平均全部** < −ε。設計卡的「k＝50% 都掉超過 ε」 |

**類別**（依序套用）：

| 類別 | 條件 | 意思 |
|:--|:--|:--|
| INSUFFICIENT_N | 題數 < 48 | 不下判定 |
| STOP | STOP | query-aware 選擇在這些任務不可靠。設計卡的停止條件 |
| SUCCESS | B 且 R | 設計卡的成功：k＝25% 補載後在 ε 內、搬運 ≤ 30%，而且 (b) 明顯好過 (a) |
| SUCCESS_NO_REFILL_NEEDED | A | 不補載就過了。設計卡的「(b) 明顯好過 (a)」不成立或不需要，所以「不載入 ≠ 丟掉」這半句沒有證據 |
| PARTIAL | B | 補載後過，但 (b) − (a) 不顯著 → 停下來問人 |
| INCONCLUSIVE | 其他 | 停下來問人 |

- 設計卡的判定＝QUEST_MEAN 的類別（`verdict_design_card`）。可部署版另列（`verdict_deployable`）。
- **兩者不同時**，例如 QUEST_MEAN＝SUCCESS、QUEST_BLIND＝INCONCLUSIVE：結論只能寫「需要問題看過完整 context 的 q 才行」，不能寫成「選擇性還原可行」。
- self-test 用合成資料走過每一個類別。

---

## 4. 和設計卡的差異與理由

| 設計卡 | 這裡 | 理由 |
|:--|:--|:--|
| Quest 式摘要選 block | QUEST_MEAN 照做，**另加 QUEST_BLIND** | C1a 的 QUEST_MEAN 用的 q 是看過完整 context 才算出來的，還原前拿不到。只報它會高估可部署的效果 |
| 補載：分數超過門檻才補 | 門檻＝當下 Quest 分數的第 m 名（排名門檻） | 我們沒有任何「decode 時 Quest 分數的尺度」的資料，絕對門檻只能憑空選。排名門檻不需要尺度。補多了會反映在位元組上，搬運 ≤ 30% 會擋掉 |
| 對照：完整還原（Cake） | FULL（遮罩路徑）＋E0_SDPA（第二階段路徑） | 判定要同一條計算路徑，避免把換路徑的雜訊算成選法的效果 |
| 對照：44e 的丟棄（ATT／QRY／位置／隨機） | 不重量 44e；同一批題目內做 (a) 的 WRITE／QRY1／POS／RND | 44e（DS2）用的也是 seed 70000 這批題，但丟棄從第 1 輪就開始、兩輪都丟，和「第 2 輪回來時才選」不是同一件事。同一個 run 裡配對比較乾淨 |
| 首 token 時間 | 還原時間〔計算〕＋Quest 額外時間〔qtime 實測〕 | 品質 run 每張卡有 3 個行程，時間欄作廢；harness 的遮罩路徑對全部 key 計算，量不出少算 |
| 工作負載 T2–T5、32K 與 128K | 32K；T5 另報、不進判定；**128K 沒做（待做）** | C1a 的 `compute_item`／`probe` 寫死 64 個 block。128K 要另寫打分、先過 128K 的 C1a 式閘門；Llama 128K 一個行程約 80–100 GiB〔估，x_common.sh〕 |
| SCBench 子集（只跑 Llama） | **不做** | 規則：不下載外部資料 |
| 選法裡的「位置」 | POS＝只載前 k%（C1a 的 NEW） | 照任務說明（Cake 前段的對照）。從後面載的 STREAM 沒做：位置帶三種輪流，前／後在期望上對稱 |
| 寫入時最好的一種 | Llama KNORM、Qwen KEYDIFF | 照 C1a 的結果。但那是**同一批題目**上選的（事後挑選）；這一格只當參考，不進判定 |
| 歷史在第 2 輪才接（第二階段 run_turns 的寫法） | 歷史在暫停前用完整 KV 接上 | 真的系統裡，第 1 輪的回答是和其他 KV 一起卸載、一起還原的（tail），不會在還原後重算。selfcheck 另外報 E0 的生成和第二階段的寫法是否相同（只給人看） |

---

## 5. 時間〔估〕與依據

**依據**（實測）：
- C1a 每題（prefill＋5 種寫入時分數＋探針，只打分）：
  - 單一行程：Llama 17.7 s、Qwen 19.3 s（`04_smoke_c1a` 的 log）。
  - 每張卡 3 個行程：一張卡 75 題 8.25 分鐘 → 每個行程約 19.8 s/題（33a：14:53:15→15:01:30Z）。
  - 本程式只算 1 種寫入時分數 → 取 ≤ 20 s。
- X2 每次生成的 `t_gen_s` 中位數（第 1 輪；每張卡 3 個 Llama／4 個 Qwen 行程；run `20261020-214139-m16-x2-*-32K-core-s*`）：

  | | T2–T4 | T5 |
  |:--|--:|--:|
  | Llama（遮罩路徑） | 0.95–1.34 s | 2.8–5.8 s |
  | Qwen | 0.70–4.72 s（依 shard） | 6.4 s |

**估法**〔估〕：每個行程（每張卡 3 個）、每題＝打分 20 s＋第 1 輪 2 s＋41 格 × 每格生成時間。

| 模型 | T2–T4 每題 | T5 每題 | 150 題合計 |
|:--|--:|--:|--:|
| Llama | 20＋2＋41×1.2 ≈ 71 s | 20＋41×4.5 ≈ 205 s | 12,660 s |
| Qwen | 22＋41×2.5 ≈ 125 s | 20＋41×6.4 ≈ 282 s | 21,105 s |

- 合計 33,765 行程·秒 ≈ 9.4 行程·小時。
- 兩張卡共 6 個行程 → **約 94 分鐘**。再加 24 次載模型（約 6 分鐘）、selfcheck＋qtime（約 10 分鐘）→ **約 1.8 小時**（兩卡並行的牆鐘）。
- 誤差抓 ±50% → 1.2–2.7 小時。

**切片**：
- 38da 的預算 105 分鐘：剩 < 10 分鐘不再領新工作，每個行程自己也照 `--budget-min` 停；正在做的那一題做完才停，最多再多約 5 分鐘。
- 沒做完的題由 38db 接（同樣 105 分鐘）。38da 全部做完的話，38db 不佔 GPU。
- 估計在 3 小時內，所以**沒有縮選法或 k**。
- 真的超過時，先犧牲的是 T5（排在每個 shard 最後）。38db 結束還有題沒做完會記 NEEDS_HUMAN；只要每個模型 T2–T4 ≥ 48 題，38dz 就能下判定。

**每張卡的行程數 S**：
- 〔估〕peak ≤ 45 GiB（C1a 冒煙實測 Llama 25.2、Qwen 39.9 GiB；本程式另外留 64 個 BF16 block：Llama 4 GiB、Qwen 0.9 GiB）。
- job 用 selfcheck 實測的 `peak_reserved_gib` 自動定：S＝min(3, ⌊176 ÷ (1.15×較大的 peak)⌋)。
- NEED-GB 150。

---

## 6. 怎麼跑

**佇列**（由排程的人放）：
- `38da_c1b.sh`、`38db_c1b.sh` 放 `queue/`。檔名排在 `39v_x3_validate`、`40_v0` 之前，不和 `38a`、`38b*`、`38c*`、`38z`、`38bz`、`38cz` 衝突。
- `38dz_c1b_report.sh` 放 `queue_c/`。
- `c1b_common.sh` **不要**放進任何佇列。

**相依**：
- 38da AFTER `33z_c1a_report`（要 C1a 的 CSV）。
- 38db AFTER `38da_c1b`。
- 38dz AFTER 兩者。
- 38da 失敗 → 38db、38dz 不會開（AFTER 要 `.done`），佇列繼續往下跑別的 job。

**工作池**（兩卡共用，`c1b_common.sh`）：
- 固定清單＝(模型, shard i/12)，兩個模型交錯：llama s0、qwen s0、llama s1、…，共 24 個。
- 每張卡 S 個 worker，各自用 `pool-claim` 領下一個還沒被領的工作。`os.mkdir` 當領取鎖，是原子操作，self-test 用 6 個行程同時搶驗證過：每個工作恰好被領一次。
- 領到的工作在自己那張卡上跑 `run --shard i/12`，跑完再領。兩張卡都會忙到工作池空。
- 該 shard 的題已全部寫完 → 標 SKIPPED，不載模型。
- 任何一個 run 失敗 → NEEDS_HUMAN、放 STOP（其他 worker 不再領新工作）、job exit 1。

**手動**（GPU 窗口；先 `source code/m7_env.sh`）：
```bash
python code/m16_c1b_restore.py --self-test                                   # CPU、不 import torch
python code/m16_c1b_restore.py selfcheck --model llama --out <json>          # GPU
python code/m16_c1b_restore.py qtime --model llama --out results/m16_stage3_mi300x/c1_selective_restore/qtime_llama.csv
python code/m16_c1b_restore.py run --model llama --shard 0/12 --budget-min 100 --out /mlsteam/data/tiara/stage3/c1b/c1b_llama_s0.csv.gz
python code/m16_c1b_restore.py pool-status --n 12                            # 每個 shard 剩幾題
python code/m16_c1b_restore.py report --raw "/mlsteam/data/tiara/stage3/c1b/c1b_*_s*.csv.gz" \
    --qtime "results/m16_stage3_mi300x/c1_selective_restore/qtime_*.csv"
```

**續跑**：重下同一個 job 即可。
- `run` 跳過同目錄任何 `c1b_<模型>_s*.csv.gz` 裡已寫完的題，所以改了切片數也不會重做。
- 寫到一半的題整題重做；report 只取同一次嘗試（同一個 run_id）的齊全列，不混用。

---

## 7. 怎麼讀

`tables_c1b.md` 每個模型、每組任務（T2–T4、T2、T3、T4、T5）一張表，每格一列：

| 欄 | 說明 |
|:--|:--|
| 答對率、對 FULL 的差 | 百分點，95% CI |
| 搬運比 | 含摘要、補載；FULLK 標 oracle |
| 答案 block 全蓋到 | 初始 → 最終 |
| 補載 | block 數、次數 |
| 同字 | 和 FULL 生成一字不差的比例 |

表後還有：
- (b) − (a) 的配對差。
- T2–T4 的判定（兩種選法各一個類別，加上每個成分）。
- 還原時間表（§2.7）。

讀的順序：
1. 看 FULL − E0_SDPA，確定換路徑的雜訊夠小。
2. 看 QUEST_MEAN 與 QUEST_BLIND 在 k＝25% 的 (a)、(b)。
3. 看答案 block 的覆蓋：答對率掉的時候，是因為沒蓋到答案，還是蓋到了也答錯。
4. 看 POS／RND：沒有資訊的選法掉多少，也就是這些任務對選擇有多敏感。
5. 看 WRITE、QRY1 (b) − (a)：補載能不能救回寫入時就選好的。
6. 最後看時間表：省下的搬運在哪個層級值得。

`c1b_summary.json` 另外記：
- `by_band2`：第 2 輪答案在前／中／後帶，各自的主格。
- `checks_false`：逐題檢查有 false 的題，含選法和 C1a 不同的。
- `contention`、`files`（含 sha256）、`qtime_run_ids`。

---

## 8. 冒煙要看什麼（38da 開頭的 selfcheck／qtime，與前幾題的 log）

### selfcheck

`$SMOKE/c1b_38da_c1b_*/c1b_selfcheck_<模型>.json` 裡每一項都要 PASS，不過 → job 自己停。

| 檢查 | 不過時 |
|:--|:--|
| 選法和 C1a 的 CSV 逐字相同（兩題、6 種選法） | 看 `per_method` 哪一種不同。可能是 GPU 不決定性，也可能是路徑差異；由人判斷 |
| forward 複本 vs KVModel（sdpa、全留遮罩、QUEST_MEAN@25 遮罩）：第一個 logits 逐位元、生成 token 相同；留 q 不改變結果 | 本程式的複本有錯 |
| k＝100% 和 FULL 逐值相同（7 種選法＋(b)） | |
| 補載的遮罩帳：沒載入的 token 數＝(64 − 已載入)×512，補的 block 在 1..63 且原本沒載入；(b) 把補載關掉＝(a) | |
| 跑完各格後，暫停前的 KV（context＋tail）逐位元沒變 | |

`info`（給人看，不設門檻）：
- `e0_same_tokens_as_stage2_run_turns`：和第二階段「歷史＋第 2 輪問題一起接」的寫法比，生成是否相同。預期多半相同；不同就記下來。
- `score_t1_full`、tail 長度。
- RND@10b 補了哪些 block：預期會補到第 2 輪的答案 block。
- `peak_reserved_gib`：job 用它定 S。超過約 50 GiB 時，S 會自動降到 2。

### qtime

`results/…/qtime_<模型>.csv`；GPU 不乾淨時不寫，只留在 run 目錄。

| 欄 | 預期 |
|:--|:--|
| `h2d_block_pinned` 的 GiB/s | 應該接近 bw.json 的 49.9／48.6。差 > 20% → 記下來，還原時間表的常數要人看 |
| `blind_q_forward`＋`quest_select_q2` | 要和省下的搬運時間比。例：CPU 層 Llama k＝25% 省 59 ms〔計算〕。比它大 → 在 CPU 層選擇性還原不划算 |
| `quest_refill_check` | 每 4 個 token 一次的額外成本 |

### run 的前幾題

每行有：每題秒數、峰值 GiB、`t1`（第 1 輪完整 KV 的分數）、FULL 與 QUEST@25 (a)／(b) 的分數、檢查不過的項目。

| 看什麼 | 預期 |
|:--|:--|
| 每題秒數 | 和 §5 的估計比：Llama T2–T4 約 70 s、Qwen 約 125 s〔估〕 |
| 工作池 | job log 裡兩張卡都在領 llama 和 qwen 的工作 |

---

## 9. 已知限制與待做（沒做的寫在這裡，不假裝做了）

| 項目 | 說明 |
|:--|:--|
| 128K | 待做，理由見 §4 |
| SCBench | 不做（不下載外部資料） |
| 逐層的 Quest（Quest 原文的做法） | 沒做。在 layer l 用當層的 q 選當層的 block、邊算邊搬。比 QUEST_BLIND 更接近原文，但要逐層傳輸、每片更小，正好撞上 I4 量到的每片固定成本。可以當下一步 |
| 混合還原 | 沒有量品質：前段用 Cake 重算（精確）＋後段只載選中的。§2.7 顯示在慢層級純載入比 Cake 慢，這個混合可能才是可用的形狀，但它的品質要另外量 |
| 首 token 時間 | 沒有端到端實測，見 §2.7 |
| T5 | 單輪，暫停點在 context 剛 prefill 完，和兩輪題的情境不同；只另報 |
| 寫入時方法 | 在同一批題目上選的（事後挑選），只當參考 |
| 品質 run 的時間欄 | 同一張卡有自己的其他行程，`t_gen_s` 全部作廢（CLAUDE.md §3 的品質規則）。分數照用，`own_gpu_intruders`、`host_level` 逐列記 |
