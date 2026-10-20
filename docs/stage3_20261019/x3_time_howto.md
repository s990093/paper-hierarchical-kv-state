# X3：壓縮組合的還原成本 × 連線速度 × GPU 忙不忙（怎麼跑、怎麼讀；第三階段，10/20 寫）

> **狀態**：程式寫好了，`--self-test` 69 項 ALL PASS（只用 CPU、合成數字＋真實常數檔的讀取；run `20261020-052345-x3-selftest`）。**GPU 路徑一次都還沒跑過**，只過了 `python3 -m py_compile` 與 symtable 的未定義名稱檢查；真正的驗證在冒煙（`03_smoke_xtime.sh`）。所以**所有結果都是 NOT_MEASURED**，`results/m16_stage3_mi300x/x3_time/` 目前是空的，也不應該有東西。
> 寫的時候第二階段正在 GPU 上量時間，所以沒有開任何 GPU 行程、沒有 import torch／vllm／transformers；CPU 指令一律 `nice -n 19 taskset -c 0-3`。
> 本頁所有時間都是〔估〕，依據寫在旁邊。本頁是 self-review，不是 cross-model review（CLAUDE.md §5）。
>
> 問題（使用者說的 B）：KV 搬到 CPU／SSD 時順便壓縮，壓多少依**連線速度**和 **GPU 忙不忙**決定。文獻與停止條件的來源：`lit/L5c_compress_on_offload.md` §5、`lit/L5d_stacking.md` §6.4（第三階段）、`lit/L5a_lowrank.md` §8 E-L3。

---

## 0. 檔案

| 檔案 | 狀態 | 內容 |
|:--|:--|:--|
| `code/m16_xtime.py` | 新 | `--self-test`（CPU）、`--smoke`（GPU）、`micro`（GPU）、`map`（CPU）、`validate`（GPU）。模組層不 import torch |
| `code/m16_xtime_kern.py` | 新 | 低秩重建的收尾 kernel（Triton：加平均＋K 套 RoPE＋寫進 KV 視圖，讀一次寫一次）＋PyTorch 參考寫法。只有 GPU 路徑 import 它 |
| `code/m16_queue/jobs/03_smoke_xtime.sh` | 新 | GPU-NEED both；`--smoke`；失敗 → NEEDS_HUMAN、exit 1 |
| `code/m16_queue/jobs/39_x3_micro.sh` | 新 | GPU-NEED both；MAX-H 1；AFTER 03_smoke_xtime；被污染自動重量一次 |
| `code/m16_queue/jobs/39z_x3_map.sh` | 新 | GPU-NEED none（cpu 佇列）；AFTER 39_x3_micro **37z_x2_report（暫定名，要核對）** |
| `code/m16_queue/jobs/39v_x3_validate.sh` | 新 | GPU-NEED both；MAX-H 2；AFTER 39z_x3_map；map 沒判值得驗證就寫原因、exit 0 |
| `/mlsteam/data/tiara/runs/20261020-052345-x3-selftest/` | 自測的 log | `stdout.log` 最後一行 `69 項；ALL PASS`；`context.txt` 記了三支程式的 md5（`20261020-052036-x3-selftest` 是改最後兩處之前的 68 項版本） |

**沒有改任何既有檔案**（`m7_model.py`、`m10_*.py`、`runner.sh`、`lib.sh`、其他 agent 的檔都沒動）。job 檔還**沒有**放進 `/mlsteam/data/tiara/stage3/jobs/`（由協調者排）。

---

## 1. 設計：要回答什麼、為什麼這樣量

L5c 的事前預測（§4）：
- **模式 Q（允許有損）**：INT4 KIVI＋一次拆在 CPU 層閒／忙都最快（第二階段 P3：Llama 33／36 ms），慢層也是 INT4 贏 → 決策圖整片 INT4，「自適應」對「事後最好的固定格式」約等於 0 → **STOP-0 很可能成立**。
- **模式 L（只准無損）**：只有 SSD 級附近，最佳的「解在哪裡」才可能隨 GPU 忙翻轉；但 GPU 無損解碼器（hipCOMP、SplitZip）沒裝（要使用者同意），所以模式 L 的動作集只剩「原樣 vs 無損＋CPU 解」，GPU 忙只能透過 H2D 在忙時變慢（×1.43–1.50）來影響決策。

L5d 第三階段要的是：疊加的組合（低秩、跨層、合併、丟 token＋量化）解碼更貴，**它們的還原成本在 GPU 忙、不同連線速度下會不會翻轉排序**。

所以 X3 分三步：
1. **micro**（GPU，量時間）：每一種「解碼段」在 duty 0／0.4／0.7／0.9 的時間，以及它把背景計算拖慢多少。
2. **map**（CPU）：把 micro 的零件放進第二階段還原器的兩條線模型，算每一層 × duty 的 32K 還原時間；報 D／F／S／A；照 §4 寫死的條件判定。
3. **validate**（GPU，量時間）：只有 map 判「值得驗證」才跑；挑 4–8 格做端到端實測，和預測比（STOP-3）。

---

## 2. micro（GPU）

### 2.1 形狀與位元組
- **載入區**＝32K 的最後 32 個 512-token block（token 16384–32767）、全層：Llama 32 層 × 8 KV head × 128；Qwen 28 × 4 × 128（`SHAPES`，程式用 config.json 交叉驗證，規則 6）。
- KV 緩衝就是 `KVModel.kv`（`[L,2,H,32768+512,D]`，和第二階段同一個排列），解碼結果寫進它的非連續視圖，和還原時一樣。
- 每個解碼段記：每 block 要搬的位元組（`bytes_blk`，實際 tensor 大小，含 scale／zero／位置）、每次還原要搬一次的位元組（`bytes_req`，例如 xKV 每個 prompt 的基底）、常駐 GPU 不搬的位元組（`bytes_resident`，例如校準 PCA 的基底）。

### 2.2 解碼段（segment）

| segment | 家族 | 內容 | ops |
|:--|:--|:--|:--|
| `bf16` | raw | 不解碼；BF16 逐 block `copy_` 進 KV 視圖（＝`Restorer._load`） | h2d_region |
| `fp8`、`int8_token`、`int4_kivi` | quant | 主線格式（`m10_codec`）。fp8 的粒度照 `fmt_choice.json`（Llama fp8_token、Qwen fp8_head）。一次拆＝`decode_multi`（第二階段 P3／S4），逐 block＝`decode_variant`（P0）；kernel 用第二階段 A 段挑的 K*（`m10_dqsweep.kstar`） | h2d_stack、dq_oneshot、dq_perblock |
| `int2_kivi`、`k4v2` | quant | 新格式（`m10_codec_x`；k4v2＝`kv-int4_kivi-int2_kivi`）。逐 block＝`decode_x_fast`；一次拆＝把 32 個 block 的 part 沿 token 軸接成一段、一次 `decode_x_fast`（KIVI 的分組是 32 個 token、不跨 block，所以和逐 block 逐位元相同，建好時驗） | h2d_stack、h2d_region、dq_oneshot、dq_perblock |
| `lr_r{16,24,32,40,64}` | lowrank | 校準 PCA 式：K、V 的係數 `[L,H,T,r]` BF16 × 每（層, head）的基底 `[r,128]`（常駐）＝`torch.matmul`；收尾（加平均、K 套 RoPE、寫進 KV 視圖）用 `m16_xtime_kern.store`（Triton 融合） | h2d_region、dq_oneshot、dq_perblock |
| `xkv_g4_f{8,4}` | xlayer | xKV 式：每 4 層共用 token 係數 `[T, r]`，`r＝4·H·128÷f`；基底 `[r, 4·H·128]` 是每個 prompt 一份 → **算進要搬的位元組**（`bytes_req`） | h2d_region、dq_oneshot |
| `merge_g{2,4}` | merge | 相鄰 g 個 token 合併（存 T/g，BF16）；還原＝展開複製 g 份（在 attention 裡等於 log g 的偏置） | h2d_region、dq_oneshot、dq_perblock |
| `droptok_k{50,25}` | drop | 每（層, head）各自留 k% 的 token（BF16）＋int16 位置；還原＝`scatter_` 進 KV 視圖＋建 key_keep 遮罩 | h2d_region、dq_oneshot |
| `lrk24_v4` | combo | K 低秩 r＝24（＋RoPE）、V 用 INT4 KIVI（＝L5a 的 K24V4，位元組比 0.25） | h2d_region、dq_oneshot |
| `droptok_k50_int4` | combo | 每 head 留 50% token → INT4 KIVI；解碼後 scatter | h2d_region、dq_oneshot |
| `merge_g2_int4` | combo | 合併 2 個 token → INT4 KIVI；解碼後展開 | h2d_region、dq_oneshot |

- **h2d_stack**＝逐 block 搬進「每 block 連續」的暫存區（＝`Restorer._load_defer`）；**h2d_region**＝逐 block 搬進「沿 token 接起來」的暫存區（strided 目的地）。一次拆吃哪一種，寫在 meta 的 `feed`。
- **bf16 不解碼**；壓縮後位元組的 pinned H2D 每個 segment 都量（就是 h2d op）。
- **丟 block（block 粒度）沒有量**：它沒有解碼成本，時間只是「少搬幾塊」，品質由 S2／DS1 決定。
- **無損＋GPU 解碼沒有量**：hipCOMP／SplitZip 沒裝，要使用者同意（§9）。無損只有「CPU 解」，用 S1 的數字（不重量）。
- 資料是亂數（時間和內容無關，第二階段同樣做法）；RoPE 用 `rope_theta` 的預設 inv_freq，**沒有套 llama3 的 rope_scaling**（只影響數值、不影響時間）。

### 2.3 計時
- 每個（segment, op）：暖機一次 → 量一次定 `inner`（一次重複至少 `--min-rep-s`＝0.1 s，跨過約 5 個 m7_busy 週期 20 ms，取平均＝期望值）→ **5 次重複**（照 I4 的 `--reps 5`），**每次重複把所有 op 的順序打散**（照第二階段 grid 的做法）。記中位數、四分位、最小、最大。
- **背景負載**：`m10_speed.Busy`（`m7_busy.py`），duty 讀 `results/m10_restore2_mi300x/busy_levels_q4.json`（idle 0、busy1 0.4、busy2 0.7、busy3 0.9；**0.9 第二階段沒有校準過重算慢化**）。每個 duty 開一個 Busy、量完所有 op。
- **背景被拖慢多少**（`micro_bg_<m>.csv`）：每個（duty>0, seg:op）開一個**新的** Busy，前景連續跑 `--bg-seconds`＝2 s；前後各一次「只有背景」當基準。前景期間的背景速度＝(背景 matmul 總數 − 基準速度 ×（總秒數 − 前景秒數))÷ 前景秒數，再 ÷ 基準速度。預設量 13 個 op（`BG_DEFAULT`）。
- **重算慢化（recompute 探針）**：同一個 process 載 KVModel，`forward_span` 在 chunk 0、8、…、56、63 各 3 次 × 每個 duty → 每個位置「忙 ÷ 閒」的中位數。map 用它把 bw.json 的閒時 f(i) 換成各 duty 的 f。
- HSA_ENABLE_SDMA：**不設**（第二階段所有量時間的 job 都沒設；ROCr 預設 1＝SDMA copy engine）。CSV 每列記 `hsa_enable_sdma`。

### 2.4 正確性與污染
- 每個 segment 建好**立刻驗**，不過就 exit 1（規則 2），不計時：量化格式和 `m10_codec(_x).decode` 逐位元相同（一次拆、逐 block 兩種都驗）；低秩／xKV 和 FP32 參考的相對誤差 <1e-2（BF16 捨入量級）、逐 block 和一次整段一致；合併和 `repeat_interleave` 參考逐位元相同；丟 token 用 `gather` 取回和來源逐位元相同、遮罩個數對。
- gpu_guard（`m10_speed._watcher`，Busy 的 pid 不算外人）包住整個 process。每列記當下的 `contaminated` 和整個 run 的 `run_contaminated`。**被污染的 run 輸出改寫到 run 目錄**（`$TIARA_RUNS/<run_id>/contaminated_micro_<m>/`），**不寫進 results**（CLAUDE.md §3），run 目錄留 `CONTAMINATED`；job 自動重量一次。

### 2.5 輸出（`results/m16_stage3_mi300x/x3_time/`）
- `micro_<m>.csv`：每（duty, segment, op）一列；`run_id`、`ts`、位元組、copy 次數、reps、inner、時間的中位數／四分位、污染。
- `micro_bg_<m>.csv`、`micro_recompute_<m>.csv`：同上，都有 `run_id`、`ts`。
- `micro_meta_<m>.json`：形狀、segment 的位元組與實作、kernel、正確性結果、Busy 的背景吞吐、torch／HIP 版本。
- `gpu_guard_<run_id>_<m>.json`。

---

## 3. map（只用 CPU）

### 3.1 還原模型（照第二階段的還原器）
- 32K＝64 個 block。**重算線** f_d(i)＝f_閒(i)（`bw.json`）× 重算慢化(d)。慢化優先用 micro 的探針；沒有時 duty 0.7 用第二階段 `summary_int4.csv` 的 compute_only 忙/閒，0.4 用第一階段 C5（單一 chunk），**0.9 沒有 → 那些格子 NOT_MEASURED**。
- **載入線**每個 block＝max(位元組 ÷ 連線, h2d_d)，逐 block 拆再加 dq_d；一次拆在載入線結束後補「dq ×載入的 block 數」（＝Restorer 的 defer：搬完才一次拆）。h2d、dq 是 micro 量到的 32 個 block 的時間 ÷ 32（**〔推算〕：每 block 成本線性**）。
- 兩條線的會合用 `m10_sweep_sim.simulate`（和 Restorer 的 claim 規則同一套）；**還原時間＝min(Cake, 全部載入)**（≈第二階段的 G5b）。
- **無損＋CPU 解**（S1）：每 block＝max(壓縮後位元組 ÷ 連線, 原始位元組 ÷ CPU 解壓速度, 原始 BF16 的 h2d)——**讀、解、H2D 完全重疊（樂觀，L5c §4.3 的假設）**。CPU 層沒有連線項。
- **vLLM 0.28 CPU 路徑**（只有全部載入，不和重算賽跑）：時間＝copy 次數 × c ＋ 64 × h2d ＋ 解碼。copy 次數＝層 × 2 × 32768÷16（vLLM block 16 token，F1 §4.6 的計法）；c 由 c6 的 32K 實測（Llama BF16 4 GiB）扣掉「同樣位元組在真實 pinned H2D 的時間」÷ copy 次數反推（〔推算〕）；**Qwen 用 Llama 的 c**；忙時的 c 乘 I4 的 foreach／scatter 忙/閒倍數（**只有 duty 0.7，而且要 I4 先跑完**；沒有就 NOT_MEASURED）。壓縮格式假設 copy 次數不變（最好的情況）。
- **已知偏差**：第二階段 09 §5 驗過，這套模擬在「慢層＋GPU 忙」偏快 23–24%；map 在這些格子的 `flags` 欄標出來。

### 3.2 層
- **名目層（判定用）**：cpu_pinned（真實 pinned H2D）、vllm_cpu、b16／b32／b48、ssd_meas、ssd_param、nfs_meas、nfs_param（出處見 §5）。
- **連線格點（只畫圖）**：L5c §5.4 的 0.25、0.33、0.383、0.83、1、3、6.98、7.87、12、20 GiB/s＋cpu_pinned。格點版本的 STOP-2 另外報（`grid_stop2`），**判定只用名目層**；兩者不一致時寫在報告裡，由人判斷。
- 慢層一律是限速器（資料在 pinned 記憶體，照位元組計時），**不是真的 SSD／NFS 讀取**。

### 3.3 三種模式（動作集）
- **L（只准無損）**：raw ∪ 無損＋CPU 解（S1 的 T4、32K、≤16 執行緒的每一種「變體｜codec｜執行緒」）。S1 沒有 → 只剩 raw → NOT_MEASURED。
- **Q（允許有損，只用第二階段過 ε 的）**：L ∪ Llama {fp8、int8_token、int4_kivi}、Qwen {int8_token、int4_kivi} × {一次拆, 逐 block}（L5c §5.1 H-B0；08 §2.3–2.5）。
- **X（疊加）**：Q ∪ X2 判定可行的 segment。**X2 沒有 → 動作集是全部（只看時間）、不下結論**。

### 3.4 X2 的輸出約定（給寫 X2 的 agent 核對）
`results/m16_stage3_mi300x/x_stack/x2_feasible.csv`，欄位至少 `model`、`x3_segment`、`feasible`、`run_id`。`x3_segment` 要用 §2.2 的名字（`seg_catalog()`）；`feasible` 是 1/0 或 True/False（＝在 ε 內）。有檔但欄位不對、或有不認得的 segment 名 → map **丟例外**（規則 7：讀不懂不等於沒有）。X2 的組合如果不在 §2.2 裡，X3 量不到它的時間——要先加 segment。

### 3.5 D／F／S／A（L5c §5.4；每個模式都報，不准只報最強的）
- **格子**＝（模型, 名目層, duty）；D 算不出來的格子不算。
- **D**＝raw（不壓縮；vLLM OffloadingConnector 只搬原始位元組）。
- **F**＝每個模型一個固定動作：在該模型所有格子的還原時間幾何平均最小者（事後挑，對 A 不利）。
- **S**＝每（模型, 層）一個動作：該層所有 duty 的幾何平均最小者（只看連線＝前作能做的）。
- **A**＝每格最好（**忙碌已知的上界**；上界都不過就不用設計預測器）。
- 好處 gain＝1 − T_A ÷ T_基準；聚合＝所有格子（兩個模型合起來）的幾何平均。另報「A 對 S ≥5% 的格子比例」「>15% 的格子比例」、F／S 的最大悔恨、每格第一名和第二名的差（<5% 在圖上打斜線）。

---

## 4. 停止條件（**看到任何數字之前寫死**；門檻在程式的 `TH`）

| 代號 | 用在 | 條件 | 結果 |
|:--|:--|:--|:--|
| **STOP-0** | 模式 Q | 每個模型以「最常勝出的格式」為主；**沒有任何一格**別的格式比主格式最好的動作快 ≥5% | 有損自適應 NO-GO（L5c 的預測：成立） |
| **STOP-1** | 模式 L | 兩個模型的 BF16 無損比（S1、T4、32K，所有變體與 codec 取最大）都 <1.15；**或**兩個模型 ≤16 執行緒的 CPU 解壓都 <0.5 GiB/s、而且沒有 GPU 解碼器 | 模式 L NO-GO；S1 沒有 → NOT_MEASURED |
| **STOP-2a** | 每個模式 | 每一格 A 對 S 都好不到 5% | GPU 忙這個軸沒有價值（退回前作） |
| **STOP-2b** | 每個模式 | A 對 F 的幾何平均好不到 5% | NO-GO（規則 4） |
| **GO_N5** | 每個模式 | A 對 S 在 ≥25% 的格子好 >15%，**而且** A 對 F 的幾何平均 >15% | 做 validate |
| **STOP-2c** | 每個模式 | 不是上面三種（介於中間） | 停下來問使用者 |
| **T-a** | 模式 X | 每一格「最快的組合」都沒有比「最快的單一方法（含 INT4＋一次拆、raw、無損）」快 ≥5% | NO-GO：疊加省下的位元組被解碼成本吃掉 |
| **T-b** | 模式 X | 組合只在 <1 GiB/s 的層贏（≥5%），在 cpu_pinned 都比最快的單一方法慢 | 寫成條件規則，不算新意 |
| **T-c** | 模式 X | 所有非量化解碼段（低秩、跨層、合併、丟 token、組合）的一次拆「忙/閒」倍數，和 INT4 一次拆的倍數差 <20%（每個 duty） | GPU 忙不改變排序，不另立成本項 |
| **T-d** | 模式 X | 有格子「最佳動作隨 duty 換人」（A≠S）而且比 S 快 ≥15% | §5.2 (c) 的新意成立；只隨連線換人另記（＝前作） |
| **E-L3** | lr_r24、lr_r40 | duty 0.7：重建「忙/閒」≥1.5 ×INT4 的倍數，**或**忙時重建 ≥50% 同位元組 H2D → 新成本項；倍數差 <20% **而且** 忙時重建 <25% H2D → 不另立項；其他 → 照實報、不下結論 | |
| **STOP-3** | validate | ≥2 格「實測 vs 預測」差 >10%；**或**實測的 A 對 F（同一種量法的配對）幾何平均好 <5% | 預測圖不可信或沒有好處 → 停 |
| 錨點警告 | map | 組成模型（micro 的 h2d＋dq）和第二階段實測（CPU 層、全部載入、E0／E1；q4 的 P0／P3、q4duty 的 P3）差 >15% | 不停，但那種動作的預測要打折看；報告裡有表 |

- 判定順序：模式 L 先看 STOP-1；STOP-2 依 2a → 2b → GO → 2c。STOP-1 成立時模式 L 的 STOP-2 標成 `STOP-1`（數字照報）。模式 X 沒有 X2 時標 `TIME_ONLY`。
- **validate 只在**模式 L 或 Q 判 GO_N5、或模式 X 有 X2 且（GO_N5 或 T-d 成立）時才跑（`flip_cells.json` 的 `worth_validate`）。
- **一定要寫進結果的**：A 對 D 的數字。即使 STOP-2 停，「不壓縮 → 每層最好的固定做法」對部署是有用的工程結論，但它**不是 B 的新意**。

---

## 5. 常數出處（map 每次都讀檔，`constants_used.csv` 會再列一次）

| 常數 | 模型 | 檔 | run_id | 怎麼來的 |
|:--|:--|:--|:--|:--|
| 閒時每 chunk 重算 f(i)（64 個） | Llama／Qwen | `results/m10_precision_mi300x/speed_{llama,qwen}/bw.json` | `20261015-154558-m10-calibf-llama`、`20261015-154647-m10-calibf-qwen` | calib-f 每 chunk 中位數 |
| BF16 每 block 位元組（和 micro 交叉驗證） | 同上 | 同上 | 同上 | `chunk_bytes` |
| 真實 pinned H2D（只當交叉參考） | 同上 | 同上 | 同上 | `real_pinned_h2d_GiBps` |
| b16／b32／b48 | 同上 | 同上 | 同上 | `targets` |
| 重算慢化 duty 0.7（備援） | Llama／Qwen | `results/m10_precision_mi300x/speed_{llama,qwen}/summary_int4.csv` | `20261016-034902-m10-grid-int4-llama`、`20261016-052143-m10-grid-int4-qwen` | compute_only CPU 32K 忙/閒 |
| 重算慢化 duty 0.4（備援） | 兩者 | `results/m7_write_policy_mi300x/busy_levels.json` | `20261008-130834-m7-c5`、`20261008-130956-m7-c5b` | 第一階段 C5 單一 chunk |
| duty 等級 | 兩者 | `results/m10_restore2_mi300x/busy_levels_q4.json` | （設定檔，沒有 run） | 13 §3.3；0.9 沒有校準過 |
| 本地 SSD 實測（ssd_meas） | 兩者 | `results/m7_write_policy_mi300x/calib_c2.csv` | `20261008-130438-m7-c2` | local read_sustained 中位數（n＝3） |
| NFS 實測（nfs_meas） | 兩者 | 同上 | 同上 | nfs read_sustained 中位數（n＝3） |
| SSD／NFS 限速參數（ssd_param、nfs_param） | 兩者 | `results/m7_write_policy_mi300x/tier_params.json` | `20261008-130257-m7-c0`、`20261008-130438-m7-c2` | 實測最慢 × 0.9 |
| vLLM CPU 路徑 32K 傳輸時間與位元組 | Llama | `results/m7_write_policy_mi300x/c6_vllm.csv` | `20261008-141324-m7-c6-vllm` | `kv_offload_total_time/bytes`（CPU_to_GPU，n＝6）中位數 |
| FP8 的粒度 | 兩者 | `results/m10_precision_mi300x/fmt_select/fmt_choice.json` | `20261015-160426-…-llama-s0` 等 4 個 | KL 最小 |
| vLLM 每次 copy 忙/閒倍數（只有 0.7） | 兩者 | `results/m16_stage3_mi300x/i4_copy_batching/copybench_sdma1.csv` | I4 的 run（**還沒跑**） | foreach、scatter、us_per_frag |
| 無損比、CPU 解壓速度 | 兩者 | `results/m16_stage3_mi300x/s1_lossless/summary.json` | S1 的 run（**還沒跑**） | T4、32K；`ratio_framed`、`decomp_MBps` |
| 品質可行集（模式 X） | 兩者 | `results/m16_stage3_mi300x/x_stack/x2_feasible.csv` | X2 的 run（**還沒跑**） | §3.4 的約定 |
| 第二階段錨點 | 兩者 | `results/m10_restore2_mi300x/q4/<m>_{int4,fp8}_P{0,3}/restores.csv`、`q4duty/<m>_int4_P3/restores.csv` | 各目錄最新的 run（RUNLOG 14.1） | E0／E1 全部載入、CPU 層的中位數 |

數字本身（例如 SSD 7.865 GiB/s、b32 1.4668）**這裡不抄**，以 map 執行當下從檔案讀到的為準（`constants_used.csv`）。

---

## 6. validate（GPU，量時間）
- 讀 `flip_cells.json`。挑格子的順序：(1) A≠S 且 A 對 S 好 ≥5%；(2) A≠F 且好 ≥5%；(3) 第一名和第二名差 <10%。同優先度裡好處大的先；最多 8 格；**不挑 vllm_cpu**（沒有還原路徑可以驗）。
- 每格量 A、S、F、D（同一個動作身兼數職只量一次）：
  - **第二階段還原器支援的**（raw；fp8／int8_token／int4_kivi 的一次拆＝S4、逐 block＝S0；int2_kivi／k4v2 的逐 block）→ 子行程跑 `m10_speed.py grid`：E0／E1、Cake＋全部載入、**2 launch × 5 rep**（L5c N5）、限速器＝那一層的頻寬、`--busy-levels busy_levels_q4.json`；同一個（模型, 格式, 方法, 層）的多個 duty 併成一次。實測＝min(Cake 中位數, 全部載入中位數)，和 map 的 min 比。
  - **其他**（低秩、xKV、合併、丟 token、組合、新格式的一次拆）→ 同一個 process 的**載入線實測**：64 個 block、全部載入（真的 H2D＋限速器＋解碼），和 map 的「全部載入」預測比（**不含重算線**，只驗載入那一半）。每項 10 次、最多 60 秒（至少 3 次）。
  - **無損＋CPU 解、vllm_cpu**：沒有還原路徑 → NOT_SUPPORTED。
- 污染的列 STOP-3 不算；還原後 KV 不對（`kv_bad`）→ exit 1。輸出：`validate/validate_cells.csv`（`run_id`、`ts`）、`validate/validate_summary.json`。

---

## 7. job 與 GPU 時間〔估〕

| job | 佇列 | GPU〔估〕 | 依據 |
|:--|:--|:--|:--|
| `03_smoke_xtime` | GPU（both） | **約 10 分鐘**（MAX-H 0.5） | 每個模型：載 KVModel 0.5–2 分鐘（權重在 /mlsteam/data）、建 4 個 block 的 segment 約 10 s、2 個 duty × 約 53 個 op × 3 次（每次 ≥5 ms）、背景 4 次 m7_busy 啟動（每次約 5–10 s）、探針 2 個 duty；validate 一次 `m10_speed grid`（4K、1 launch × 1 rep，含載模型約 2 分鐘）＋一格載入線 |
| `39_x3_micro` | GPU（both） | **每個模型 12–15 分鐘，合計約 0.5 小時**（MAX-H 1） | 計時：約 53 個 op × 5 次 × ≥0.1 s × 4 個 duty ≈ 2 分鐘；背景拖慢：3 個 duty × 15 次新的 m7_busy（啟動 3–5 s＋前景 2 s＋收尾暫停 1–5 s）≈ 7–9 分鐘，**這是大頭**；探針 4 個 duty × 27 次 forward_span ≈ 1 分鐘；建 segment（編碼＋pin 約 12 GiB）約 1 分鐘；載模型 0.5–2 分鐘。被污染重量一次最多再加一倍 |
| `39z_x3_map` | cpu | 0（CPU 幾秒到 1 分鐘） | 兩個模型 × 9 層 × 4 duty × 約 60 個動作的還原模擬（純 Python）；自測整條 map 在合成資料上 <1 s |
| `39v_x3_validate` | GPU（both） | **0.5–1.5 小時**（MAX-H 2）；map 沒判值得驗證就 0 | 每組 `m10_speed grid`：2 launch 各載一次模型（約 1 分鐘）＋capture＋E0／E1 × 2 種模式 × 5 rep；NFS 級一次 BF16 全部載入約 10 s → 每組 3–8 分鐘，最多約 10–12 組；載入線每項最多 60 s |

量時間的 job 都是兩卡獨佔（runner 的 timing 角色，卡 1），排在第二階段之後（`S3_AFTER_S2=1`）。

---

## 8. 怎麼跑、怎麼讀

```bash
# 自測（CPU、不 import torch）
nice -n 19 taskset -c 0-3 python -I code/m16_xtime.py --self-test
# 冒煙（GPU；由 03_smoke_xtime.sh 跑）
python code/m16_xtime.py --smoke --out /mlsteam/data/tiara/stage3/smoke/x3
# 正式
python code/m16_xtime.py micro --model llama --out results/m16_stage3_mi300x/x3_time
python code/m16_xtime.py micro --model qwen  --out results/m16_stage3_mi300x/x3_time
python -I code/m16_xtime.py map --micro-dir results/m16_stage3_mi300x/x3_time --out results/m16_stage3_mi300x/x3_time
python code/m16_xtime.py validate --map-dir results/m16_stage3_mi300x/x3_time --out results/m16_stage3_mi300x/x3_time/validate
```

**讀結果的順序**：
1. `map_report.md` 最上面的「判定」：STOP-0（模式 Q）、STOP-1（模式 L）、每個模式的 STOP-2 代號、T-a～T-d、E-L3、validate 跑不跑與原因。
2. 「組成模型對第二階段實測」表：組成（micro 的 h2d＋dq ×64）和第二階段的 P3／P0 全部載入差多少；>15% 的打 ⚠——那種動作的預測先打折看。
3. 「聚合」表：每個模式的 A 對 D／F／S。**A 對 D 一定要寫進結果**（部署價值），但不是新意。
4. `decision_<模式>_<模型>.svg`：列＝層（快 → 慢），欄＝duty；顏色＝最佳動作的家族；斜線＝前兩名差 <5%；灰＝NOT_MEASURED。**如果每一列的顏色不隨 duty 變（垂直不變），GPU 忙這個軸就沒有東西**（L5c §5.6）。
5. `map_decision.csv`：每格 A、第二名、S、F、D 與好處；`flags` 欄＝近乎平手。`map_cells.csv`：每個（格子, 動作）的還原時間、Cake 或全部載入、會合點 b、`flags`（慢層＋忙偏快、無損的重疊假設、vLLM 的 c 怎麼來、重算慢化的來源）。
6. validate 跑了的話：`validate/validate_summary.json` 的 STOP-3 與 `bad_cells`。

---

## 9. 限制與要人決定的事

1. **GPU 無損解碼器（hipCOMP-core、SplitZip 的 Triton kernel）：待使用者決定**。L5c §5.7：要從原始碼建／移植，要使用者同意；這裡**沒有裝、沒有編**。沒有它，模式 L 的「在哪裡解」只有 CPU 解 vs 原樣，B2 的 GPU 忙翻轉只能透過 H2D 變慢體現。
2. **`37z_x2_report` 是暫定的 job 名**：寫 X2 的 agent 決定真正的名字；對不上的話 `39z_x3_map` 永遠不會開（runner 的 AFTER 等不到 `.done`）。X2 的輸出要照 §3.4 的約定，否則 map 會丟例外。
3. **vllm_cpu 在忙時要 I4（21_i4_sdma1）先跑完**；只有 duty 0.7。沒有 I4 → 那些格子 NOT_MEASURED（不是 0）。
4. **duty 0.9 的重算慢化只有 micro 的探針**；探針被污染或沒跑 → duty 0.9 的格子整列 NOT_MEASURED。
5. **一次拆的成本線性外推**（32 個 block 量 → 64 個 block 用）；載入線實測（validate）會抓到大的偏差。
6. **無損的三段完全重疊是樂觀的**（L5c 同一個假設）；validate 沒有無損的還原路徑可以驗。
7. **validate 的 harness 子行程把 restores.csv 寫進 results**（和第二階段一樣，污染列有 `contaminated_gpu` 欄、STOP-3 不算它們）；micro 被污染則整個不進 results（§2.4）。
8. 低秩的收尾 kernel 是自己寫的 Triton（`m16_xtime_kern.py`）；Triton 失敗時**不會自動退回** PyTorch 寫法（規則 2），由人決定要不要用 `--store-impl torch` 重跑（會讓低秩看起來更貴）。
9. 冒煙時要特別看的地方，見回報（Triton 收尾 kernel、`scatter_` 對非連續視圖、新格式一次拆的逐位元比對、m7_busy 收尾時的 gpu_guard、validate 子行程的 run_id 對應）。
10. **和 X2（`code/m16_xstack.py`，另一個 agent 寫的）的對應還沒接上**：10/20 05:2xZ 看到它用組合語法（`drop:<方法>:<比例>`、`merge:<g>`、`lr:k<r>[:v<r>]`、`xl:k<r>…`、`q:<格式>`，用 `+` 串），輸出是 `x2_selection.json`／`x2_report.json`，**沒有** §3.4 的 `x2_feasible.csv`。所以現在 map 會把模式 X 當成「只看時間」。要接上有兩條路（要人決定）：(a) X2 的 gen-report 另寫一份 `x2_feasible.csv`，把它的組合對到 §2.2 的 segment 名；(b) X2 選出組合之後，在 X3 加對應的 segment（例如 `lr:k24:v0+q:int4_kivi`＝`lrk24_v4`；係數再量化的低秩、block 粒度丟＋量化目前 X3 沒有對應的段）。block 粒度的丟沒有解碼成本，只是少搬幾塊——可以由量化段的每 block 成本組出來，目前 map 沒有做。
