# A0（＝G-M01/02）：縮圖還原的上界——「全部讀粗底、新問題挑少數塊補細節」能不能贏同位元組的均勻壓縮

> **性質**：判準登記（事先寫死）。登記時間見文末「登記」一節（`date -u`）。登記之後這一節以上不改；和登記不同的做法寫成文末的「偏離 N」。
> **來源**：[05 §4](../05_poc_plan_h200.md)（A0 的配置與判準 GA0-*）、[mechanisms/06 §1–§2](../mechanisms/06_eval_gates.md)（主設定改 128K／256K、加 14B、加 ONE-PASS 的閘）、[mechanisms/02](../mechanisms/02_bytes_quality.md) M01／M02。
> **平台**：只在 H200（hgx-cn01）。品質實驗：不量時間，可以和別人的 job 同時跑；照使用者 10/25 的排程規則不和 ICPE 量時間的 job 重疊（memory `stage3-gpu-scheduling`）。
> **這一關回答**：在讀回同樣多位元組的前提下，「每塊都讀 16–32 倍的粗底、只把新問題挑中的少數塊升級成 INT4 或 BF16」，答對率能不能比「每塊同樣壓縮」高 ≥2 個百分點。先用知道答案的挑法（ORACLE）看上界；上界不夠大就停（GA0-STOP）。

---

## 1. 設定

### 1.1 模型、長度（判定格）

| 判定格 | 模型 | 長度 | n_blocks（×512） | 區內 block（nreg） |
|:--|:--|:--|--:|--:|
| L128 | Llama-3.1-8B-Instruct（`4699cc75…`） | 128K | 252（129,024 token） | 251 |
| Q128 | Qwen2.5-7B-Instruct-1M（`e28526f7…`） | 128K | 252 | 251 |
| Q256 | Qwen2.5-7B-Instruct-1M | 256K | 504（258,048 token） | 503 |
| Q14-256 | Qwen2.5-14B-Instruct-1M（`620fad32…`） | 256K | 504 | 503 |
| （對照列，不判定） | Llama、Qwen-7B | 32K | 64 | 63 |

- 長度和 3a X4c（128K＝252 塊）、X4 的 Qwen 256K 切片（504 塊）相同；128K 留 2,048、256K 留 4,096 個位置給尾段、問題與生成（Llama 上限 131,072；Qwen noDCA 上限 262,144）。
- harness 是 `m7_model.KVModel`（HF 權重、自管 KV；HF 的 Qwen2 沒有 DCA，所以 1M 模型在 harness 裡就是 noDCA，和 X4 的 Qwen 256K 相同）。
- 不開 YaRN（CLAUDE.md §8）。
- **attention**：context 的 prefill 用 fa3（ICPE P0-1 S1：五個候選都過正確性閘、fa3 最快）；之後所有的 forward（打分、teacher forcing、生成）一律 sdpa。所有配置共用同一份 prefill 的 KV，E0 也是 sdpa 接在同一份 KV 上，所以配置之間的比較不受 prefill 用哪個 kernel 影響。
- Qwen-14B 的題目參數（m10_tasks 的難度）沿用 Qwen-7B 的（`CANDIDATE_PARAMS.json` 的 `qwen`）。Qwen-7B 和 14B 的 `tokenizer.json` 是同一個 git blob（`443909a6…`），題目逐 token 相同；開跑時再用 tokenizer 本身核對一次（不同就停）。

### 1.2 題目（m10_tasks；T2–T5，T5＝cwe 是「彙整」類，T2–T4 是「找答案」類）

| 用途 | seed0 | 每任務題數 | 合計 | 用哪一輪 |
|:--|--:|--:|--:|:--|
| TF 篩選 | 71000 | T2–T5 各 24 | 96 | 第 1 輪（單輪） |
| 生成（判定用）單輪 | 72000 | T2–T5 各 48 | 192 | 第 1 輪 |
| 生成（判定用）第 2 輪 | 73000 | T2–T4 各 32 | 96 | 第 2 輪（第 1 輪在暫停前用完整 KV 做完） |

- 題目的排法＝`m16_xstack.plan_interleaved`（T2–T4 依 ROTATIONS 輪流、T5 單輪；seed＝seed0＋k）。三組 seed 和所有既有題組（60000–67047、70000–70044、40000–40011）不重疊（`code/s3/a0.py plan` 會檢查，重疊就停）。
- **生成的判定格合計 288 題**（05 §2：每個判定格 ≥288 題）。TF 和生成用不同的題，TF 挑出的配置不在判定的題上挑。
- 第 2 輪的情境照 C1b：context prefill 完 → 第 1 輪問題＋第 1 輪的標準答案歷史（`history_ids`）用完整 BF16 KV 接上（＝暫停前的快取）→ 暫停。回來時只有 context 的區內 block（1…nb−1）被壓縮；block 0 與尾段（第 1 輪問題＋歷史）一律 BF16。新問題＝第 2 輪的 `query_ids`。

### 1.3 讀回的表示法

**區內**＝block 1…nb−1。block 0（chat 開頭、任務說明）與尾段在所有配置都讀 BF16、不算進區內位元組（和 X4 的 `ratio_region` 同一個口徑；報告另列含 block 0 的總比例）。

**粗底**（每塊都讀）：xKV 式跨層低秩（W＝4 個相鄰層一個窗口），基底**事先校準**（`:cal`，校準題＝`m16_xstack` 的 calib 題組 seed 61000、T2–T5 各 2 題、32K），係數量化。基底跨請求共用、不算進每個請求的位元組（`predict_bytes` 的 `shared.*`）。校準照 `m16_xstack calib-xl` 原樣（不去平均、K 先逆 RoPE），只把保存的秩上限從 64 提到 96（「每 head 每層等效秩」；同位元組配對的 rv 最大到 96），前 64 個特徵向量和上限 64 時相同。秩照「倍數 ≥ T 的最大 rk、rv＝round(1.5·rk)」（K:V＝1:1.5，X4 Q-A 的結論）；係數格式：T ≤ 20 用 `int4_kivi`，T ≥ 24 用 `kv-int4_kivi-int2_kivi`（X4 Q-A：低於 4 bit 在 24×、32× 值得）。倍數只和 rk 有關，三個模型相同：

| 名稱 | spec | 倍數 | 區內位元組比 f |
|:--|:--|--:|--:|
| B16 | `xl:k20:v30:cal+q:int4_kivi` | 16.38 | 0.0611 |
| B24 | `xl:k17:v26:cal+q:kv-int4_kivi-int2_kivi` | 25.13 | 0.0398 |
| B32 | `xl:k13:v20:cal+q:kv-int4_kivi-int2_kivi` | 32.77 | 0.0305 |

**精修**（只讀挑中的塊，整塊取代粗底；粗底那一份照樣算位元組＝保守的 simulcast 口徑）：
- INT4＝`int4_kivi`（3.2×，f＝0.3125）；BF16＝原值。
- 升級比例 ρ ∈ {1/16, 1/8, 1/4}，挑中的塊數 m＝round(ρ·nreg)（nreg＝251 → 16、31、63；503 → 31、63、126；63 → 4、8、16）。

### 1.4 挑法（新問題＝要回答的那一輪的 `query_ids`）

打分一律是同一個式子：對每層 ℓ、每個 q head h、每個問題 token t，在 context 的 key（[0, ctx_end)：block 0＋區內＋尾段，不含問題自己的 key）上做 softmax(q·k/√D)，把落在區內 block j 的機率加起來：

- **全域**（`gran=global`）：a_j＝Σ_{ℓ,h,t} → 每塊一個分數，前 m 名在**所有層**升級。
- **逐層**（`gran=layer`）：a_{ℓ,j}＝Σ_{h,t} → 每層各自前 m 名（每層讀的位元組和全域相同）。

| 挑法 | q 從哪來 | 打分用的 K | 可部署 | 額外讀的位元組 |
|:--|:--|:--|:--|:--|
| ORACLE（global）／ORACLE-L（layer） | BF16 KV 上跑問題（精確） | BF16 原值 | 否（上界） | 0 |
| EXACT-Q | 同上（精確的 q） | 粗底解碼後的 K | 否 | 0 |
| BASE-ATTN（global）／BASE-ATTN-L（layer） | 粗底 KV 上跑問題（兩趟：第一趟打分、組好 KV 後第二趟重跑問題） | 粗底 | 是 | 0 |
| ONE-PASS（layer） | 一趟：第 ℓ 層的 q 來自已經組好的第 1…ℓ−1 層；第 ℓ 層在粗底上打分、挑完把這層挑中的塊換成精修，再算這層的 attention | 粗底 | 是 | 0 |
| QUEST-512 | BASE-ATTN 第一趟的 q | 每塊、每（層, KV head）逐維 min／max（BF16，寫入時存）；彙總照 C1a 的 QUEST_MEAN（略過前 2 層） | 是 | 每塊 2·L·H·D·2 B |
| KSCAN-2b | BASE-ATTN 第一趟的 q | 逐 token 2-bit K（`int2h_g32`） | 是 | 區內 K 的 `int2h_g32` 位元組 |
| WRITE-TIME | 不用問題 | 寫入時分數：Llama KNORM、Qwen／Qwen-14B KEYDIFF（C1b 的選擇），`m10_drop.unit_order` 的彙總 | 是 | 塊遮罩 ⌈nreg/8⌉ B |
| RANDOM／RECENT／SEQ | 不用問題 | 隨機（每題固定種子）／最後的塊先／最前的塊先（＝「全部依序精修到期限為止」） | 是 | 0 |

### 1.5 TF 篩選的配置（每題都跑；每題約 188 個）

1. **E0**：不壓縮。
2. **均勻（固定格點）**：`q:int4_kivi`（3.2×）、`q:int2_kivi`（5.33×）、U8 `xl:k40:v60:cal+q:int4_kivi`、U12 `xl:k27:v40:cal+q:int4_kivi`、U16＝B16、U20 `xl:k16:v24:cal+q:int4_kivi`、U24＝B24、U32＝B32。
3. **均勻（同位元組配對）**：對 §1.3 的 18 個 (粗底, ρ, 精修) 位元組水準 f_P，各配一個「位元組 ≤ f_P 的最大者」：候選＝`xl:k{rk}:v{round(1.5rk)}:cal`（rk 4…64）× {int4_kivi, kv-int4_kivi-int2_kivi} ∪ {`q:int4_kivi`, `q:int2_kivi`, `q:int8_token`}。這就是 05 §4.2 的「同碼、查詢無關」對照（同一個碼、每塊讀一樣深）。
4. **漸進（核心）**：{B16, B24, B32} × ρ{1/16, 1/8, 1/4} × 精修{INT4, BF16} × 挑法{ORACLE, ORACLE-L, EXACT-Q, BASE-ATTN, BASE-ATTN-L, ONE-PASS}＝108。
5. **漸進（其他挑法）**：{QUEST-512, KSCAN-2b, WRITE-TIME, RANDOM, RECENT, SEQ} × {B16, B32} × 3 個 ρ × 精修 INT4＝36。
6. **丟塊對照**（＝C1b，沒挑中的塊不用，key_keep 遮掉）：挑法{ORACLE, ORACLE-L, BASE-ATTN（B32 的第一趟）} × 3 個 ρ × 精修{INT4, BF16}＝18。

TF 的量：答案每個 token 的 log 機率（`m10_sweep.Item.metrics`：gold_lp、kl、top1、first_ok），Δgold_lp＝配置 − E0（同題配對）。

### 1.6 從 TF 挑生成配置（每個判定格分開挑；寫死）

- G1：ORACLE 與 ORACLE-L 的配置裡，區內 f ≤ 1/12 的，依 TF 平均 Δgold_lp 由大到小取前 3 個不同的 (粗底, ρ, 精修)（平手取 f 小的）。
- G2：這 3 個 (粗底, ρ, 精修) 各帶：ORACLE、ORACLE-L、BASE-ATTN、BASE-ATTN-L、ONE-PASS、它的同位元組均勻配對；TF 第一名的那一個再帶丟塊對照（ORACLE、精修同一個）。
- G3：錨點：E0、`q:int4_kivi`、TF 平均 Δgold_lp 最好的均勻配置（f ≤ 1/12）。
- G4：同長度另一個判定格（128K：L128 ↔ Q128；256K：Q256 ↔ Q14-256）的 G1 裡、這一格的 G1 沒有的 (粗底, ρ, 精修)：加它的 ORACLE 和同位元組配對（GO 的「另一個模型方向相同」要在同一個配置上比）。所以**同長度兩格的 TF 都做完才挑**；另一格量不到（NOT_MEASURED）時照實記，那個長度不能判 GO。
- 合計最多 3×6＋1＋3＋3×2＝28 個；重複的只跑一次。
- 32K 對照列只跑 TF，不生成。

生成：greedy，長度照主線（T5 120、其他 64；`m10_quality.gen_len`），計分＝`m10_tasks.score`。漸進配置的挑塊在問題 prefill 時做一次（ORACLE／BASE-ATTN：第一趟打分 → 組 KV → 重跑問題 → 生成；ONE-PASS：一趟問題 prefill 同時挑塊 → 生成），生成途中不改。

### 1.7 位元組帳（逐列記）

bytes_base（粗底，區內全部塊）、bytes_refine（m 塊 × 精修格式的每塊位元組；逐層時＝每層 m 塊 × 每層每塊的位元組，總量相同）、bytes_index（§1.4 最後一欄）、bytes_region_bf16（nreg 塊的 BF16 位元組）。**f＝(base＋refine＋index) ÷ region_bf16**。丟塊對照沒有 base（只有 refine＋index＋遮罩）。均勻配置的位元組由實際的編碼輸出加總，並和 `predict_bytes` 逐項比對（不同就停，規則 2）。

---

## 2. 判準（事先寫死；判定用生成答對率，以題目為單位配對 bootstrap 10,000 次、95% CI；ε＝2 個百分點）

令一個判定格裡、一個漸進配置 P（ORACLE 或 ORACLE-L）的區內位元組比 f_P；它的**最好均勻對照** U*(P)＝生成集合裡區內 f_U ≤ 1.05·f_P 的均勻配置中，生成答對率最高的那一個。

- **GA0-GO**：存在判定格 c 與配置 P，使得
  1. f_P ≤ 1/12；
  2. P 對 E0 的配對差 CI 下界 ≥ −2 個百分點；
  3. P 對 U*(P) 的配對差點估計 ≥ +2 個百分點，且 CI 下界 > 0；
  並且**另一個模型**在同長度的判定格（128K：L128 ↔ Q128；256K：Q256 ↔ Q14-256）上，同一個 (粗底, ρ, 精修, 挑法) 對它自己的 U* 點估計 > 0。
- **GA0-STOP**：沒有任何 (c, P) 滿足條件 3 的點估計 ≥ +2 → 方向 A 停，寫成負面結果（「在 16–32 倍的範圍內，KV 的重要性不夠集中，挑塊升級拿不回均勻壓縮的損失」）。
- **GA0-WEAK**（兩者之間：有條件 3 但其他條件沒過，或另一個模型方向相反）→ 不判 GO，照實報告，11/17 跟老師討論。
- **GA0-SEL**（只在 GO 時看；用使 GO 成立、生成答對率最高的 P）：可部署挑法拿到 ORACLE 好處的比例 s＝(acc(D)−acc(U*))／(acc(P)−acc(U*))，D 取 BASE-ATTN、BASE-ATTN-L、ONE-PASS 中最高的（三個都報）。s ≥ 0.7 → 不需要學的挑塊器；0.5 ≤ s < 0.7 → 做 A-DL1；s < 0.5 → 先做 D0（原生索引），不行就停。
- **GA0-DROP**：同一個 P 的 (ORACLE, ρ, 精修)，「變糊」（漸進）對「丟掉」（丟塊對照）的配對差 ≥ +5 個百分點 → 「不丟」本身有貢獻。
- **G-M02（逐層消掉 query 誤差）**：在 G1 的 3 個 (粗底, ρ, 精修) 上，以 ORACLE-L 為準：gap_ONE＝acc(ORACLE-L)−acc(ONE-PASS)、gap_BASE＝acc(ORACLE-L)−acc(BASE-ATTN-L)。3 個裡 gap_ONE < gap_BASE（點估計）的少於 2 個 → 「逐層消掉 query 誤差」不成立（機制仍可用，但只剩省時間）。報 gap_BASE−gap_ONE 的配對 CI。
- 丟塊對照走 key_keep 的遮罩路徑（`KVModel._attn_keep`），E0 與漸進配置走沒有遮罩的 sdpa；兩條路徑的差在 C1b 是 +0.0 [+0.0, +0.0]（FULL − E0_SDPA）。這一關不另外量路徑本身的差，報告裡寫明。
- **另報（不判定）**：每個挑法在 TF 上的 Δgold_lp–f 前緣；答案 block 被挑中的比例（逐層時對層平均）；「找答案」（T2–T4）與「彙整」（T5）分開；單輪與第 2 輪分開；EXACT-Q 對 BASE-ATTN 的差（＝「query 不精確」的損失，L8a §0 第 6 點：沒人量過）；32K 對照列。

---

## 3. 量法、規則

- 每題：fa3 prefill → 切 sdpa → E0 的 TF（或生成）→ 每個配置組 KV（只改和上一個配置不同的（層, block））→ 量。組 KV 的正確性：全部指向原值的組合要和 E0 逐位元相同；ONE-PASS 的 forward 在不挑任何塊時要和 `KVModel.forward_span` 逐位元相同（gpu-selftest）。
- GPU 監控：`GpuWatcher`（m10_quality 的品質規則：污染時分數保留、逐列記爭用、`t_*` 時間欄作廢）。
- 分片：每個 job ≤4 小時、逐題續跑（同一組輸出檔裡做完的題不重做）；`--budget-min` 到了就在題與題之間停。
- 輸出：逐列 CSV.gz（每列有 `run_id`、`ts`）在 `results/s3_h200/a0/raw/`（>1 MB 的照 3090 慣例 gzip）；摘要與判定在 `results/s3_h200/a0/`。
- 失敗照實記、停下回報；不換方法硬幹（CLAUDE.md §1）。

## 4. 成本〔估計，NOT_MEASURED〕

TF：L128 約 1.5、Q128 約 1、Q256 約 4、Q14-256 約 8 GPU 小時；生成：約 4、3、7、13 GPU 小時；32K 對照約 0.5。合計約 40–50 GPU 小時（比 06 §2 的 15–25 多：加了 14B、256K、每格 288 題）。第一次冒煙之後用實測的每題時間重估。

## 5. 和 05／06 不同的地方（登記前就決定；理由）

1. **K2 家族（逐 token 2-bit K＋低秩 V）不當粗底**：2-bit K 加上 scale 一個值 3 bit，光 K 就佔 BF16 的 3/32，K2 家族的倍數上限約 10×，做不到 05 要的「對齊 16／24／32×」，也不可能滿足 GA0-GO 的 f ≤ 1/12。改成 KSCAN-2b 這個挑法（2-bit K 只當索引、位元組照算），回答「粗底只有低秩 K 時挑塊準不準」（2604.08426 §4.3 的疑慮）。
2. **粗底用事先校準的基底（`:cal`）**，不用每題 SVD：可部署、和 W0 的寫入端假設一致；3a 38k 校準 vs 每題 SVD 分不出差（Llama +0.4 [−1.7, +2.6]）。代價：和 X4c（每題 SVD）的 128K 數字不能逐列比。
3. **ONE-PASS、ORACLE-L、BASE-ATTN-L** 是 06 §2 的 G-M01/02 加的；逐層的兩個是為了讓 G-M02 的比較只差「q 從哪來」一件事。
4. **QUEST-512 的 q** 用 BASE-ATTN 第一趟的（還原當下 GPU 上真的有的東西），不是 C1b 的 QUEST_MEAN（看完整 context，偏樂觀）。
5. **生成題數** 192 單輪＋96 第 2 輪＝288（05 寫 192＋150；改成剛好達到 05 §2 的 ≥288，省 256K 的時間）。
6. **32K 只當對照列、只跑 TF**（06 §1）。
7. **時間不在這一關量**（A2 才量）；這一關只量品質與位元組。

---

## 登記

- 登記時間（`date -u`）：**2026-10-25T12:23:54Z**；commit：見 `git log -- docs/stage3_design_20261024/runs/A0_progressive_restore.md` 的第一個 commit（commit 訊息開頭 `docs(s3-a0): prereg`）。
- 登記時還沒有任何 A0 的數字：模型權重還在下載（job 22235 排隊中），沒有跑過任何 GPU 步驟。
- 程式：`code/s3/a0.py`（CPU 自測 PASS）、`code/s3/a0_gpu.py`（還沒在 GPU 上跑過；gpu-selftest 是第一步）。計畫檔：`results/s3_h200/a0/plan_<cell>.json`。
