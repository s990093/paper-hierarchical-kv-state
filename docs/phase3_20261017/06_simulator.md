# 第三階段 統一模擬器（`code/p3_sim/`）

> **狀態**：工具與驗證（2026-10-18，統一模擬器 agent）。**沒有跑完整地圖**：完整地圖要等 `04_prereg_map.md` 凍結。
> 規格：`04_prereg_map.md` v2（§1 壓縮決定、§2 E1–E6、§3 C0–C4 與 O、§4 指標、§5 場景軸、§8 限制）。
> 只用 CPU（`TIARA_NO_GPU=1 bash code/run3090.sh p3sim-<短名> …`）。冒煙的數字只放在 run 目錄，不進 `results/`。
> 沒有 commit（照指示）。既有的 `code/p3x/*.py`、`code/m7_sim.py`、`code/p3_data/` 一行都沒改，只 import。
> 這份文件是邊做邊寫的：每完成一塊就補一節。

---

## 1. 架構

| 檔 | 內容 |
|:--|:--|
| `build_inputs.py` | 從 repo 的實測檔產生 `config/costs_mi300x.json`、`config/costs_3090.json`、`config/format_table.PLACEHOLDER.json`（run `20261018-002806-p3sim-inputs`） |
| `config/outer_policies.json` | 外層做法 E1–E6 的參數與 X6 的出處（§2） |
| `formats.py` | 格式表（大小、解壓、品質）與平台常數（f(i)、GPU 忙倍數、CPU 層、SSD 層）的載入；NOT_MEASURED 用到就丟 `NotMeasured` |
| `policies.py` | 壓縮決定 C0–C4、O、學出來的預測器的介面（§3） |
| `trace.py` | 統一資料表 → 引擎輸入 P（和 `x1_sim.prep` 同形）；切窗、以 session 為單位抽樣、疊加（半合成）；規則 6 的斷言（§5.1） |
| `engine.py` | **session 引擎**（serve=trace）：X1 的模型（照抄 `x1_sim.Engine` 的算式）＋格式＋E1–E5＋逐筆 I/O 紀錄。放置和時間無關 → 一次跑很多時間設定（View） |
| `engine_block.py` | **block 引擎**（serve=fifo1）：X4 的模型（照抄 `x4_sim.Sim.run_events` 與增量版 QP）＋格式＋記帳。給 E6 與它同引擎的 E2／E1 基準 |
| `ledger.py` | 記帳：1 秒窗口、尖峰 p99、總量、品質帳、TTFT、截尾版、記帳恆等式（§4） |
| `sim.py` | 跑一格：`run_cell`（session 引擎，多個時間設定）、`run_cell_fifo1`（block 引擎，單一時間設定） |
| `smoke.py` | 小樣本冒煙（§7） |
| `timing.py` | 量每個請求要多久（§9 的估計） |
| `test_p3_sim.py` | 不變量測試 (i)–(v)＋記帳恆等式＋和 X8 的成本模型對拍（§6） |

**兩個引擎的分工**（為什麼有兩個）：04 §8 說模擬器一次服務一個請求、排隊只在 E6 粗略模擬。X1（session 層級 LRU、refcount 共用前綴、不排隊）
和 X4（block 層級 LRU、一張 GPU FIFO、QP）的儲存模型不同，要各自逐位元重現（不變量 (i)、(ii)），只能各留一份算式。
**E6 的「省多少」必須在 block 引擎內和它自己的 E2 比**（同一個儲存模型），不能拿 session 引擎的 E2 當基準。

### 1.1 輸入常數（`build_inputs.py`，run `20261018-002806-p3sim-inputs`）

全部從 repo 裡已有的實測檔照既有規則抽出來，每個值在 JSON 裡附來源檔與來源 run：

| 常數 | MI300X | 3090 |
|:--|:--|:--|
| f(i)（Llama） | `x1_sim.constants()`＝`m8_trace_oracle.load_f`（calib_c1.csv 的 C1 中位數，i≥80 線性外插）——**和 X1 同一份**，所以能逐位元對 X1 | X7 `f_ms_idle_32k`（64 個，之後線性外插） |
| f(i)（Qwen） | `speed_qwen/bw.json` 的 `f_ms_median_by_chunk`（x5_costs 的規則） | X7 |
| GPU 忙倍數 | `busy_levels.json` busy2＝2.07（X1／X8 用的；Qwen 自己的 x5 規則值另記，只報告） | X7 busy 0.7（Llama 1.906、Qwen 1.913） |
| CPU 層 | `tier_params.json` cpu（讀 35.4、寫 35.5 GiB/s、k 1.161） | 讀＝chunk÷ell_cpu_ms（Llama 6.99、Qwen 6.91 GiB/s），寫＝c0 D2H 6.32，k＝duplex 1.199 |
| SSD 層（設定檔 `ssd_profiles`） | `mi300x_local`（讀 6.98、寫 1.91、k 3.306）、`nfs`、`b32` | 同 `mi300x_local`（04 §5 主地圖的 SSD）；`ssd8_3090`＝NOT_MEASURED |
| CPU 忙（H2D 變慢倍數） | NOT_MEASURED → 模擬器拒跑 cpu=busy | NOT_MEASURED（等 3090 量） |
| 70B 級 | f(i) NOT_MEASURED → 只算頻寬、還原一律 load-only、TTFT 不算 | 同 |

### 1.2 格式表：placeholder

- 正式表由另一個 agent 做（`results/p3_formats/format_table.json`；它的結構是「每列＝模型×格式」，和這裡的不同）。
  **它出來之前，模擬器只讀 `config/format_table.PLACEHOLDER.json`**，檔頭寫 `"PLACEHOLDER": true` 與警告；
  載入函式預設就是 placeholder，**不會因為正式表出現就悄悄換掉**。正式表出來後要寫一個轉換函式接進 `formats.load_format_table`（不改引擎）。
- placeholder 的內容：
  - 每 block 位元組：bf16＝kv_bytes_per_token×512（架構）；8bit／int4＝第二階段 restores 的 median(bytes_stored/n_blocks)（X5 的規則，附 run）；
    int2、Qwen 的 int8、70B 的低精度＝NOT_MEASURED（沒有大小的格式，模擬器不准用）。
  - 解壓（每 block）：MI300X＝X1 的規則（E1、load_only、cpu、median(t_dequant/fp8_read)，附 run）；3090＝X7 的還原中解壓（只有 Qwen）。
  - 品質：bf16＝pass（參考），其他**一律 unverified**。所以用 placeholder 跑出來的品質帳沒有任何意義，只是確認程式會記帳。
- **測試不讀 placeholder**：每個測試自己在程式裡造一份格式表（數值是測試用的，和真實無關），見 §6。

---

## 2. 外層做法 E1–E6（`config/outer_policies.json`）

外層做法是「場景」，不是我們的決定。參數照 X6 查到的各系統預設（X6 §4.1，permalink 代號見 X6 附錄）。

| 代號 | 模擬器做的事（`used`） | 代表（X6） | 查到但沒模擬的 |
|:--|:--|:--|:--|
| E1 | 只有 CPU；滿了把最久沒用的 session 整個丟掉，回來時沒命中的部分重算（X1 的 `LRU+NONE`） | vLLM 0.28 OffloadingConnector 單層：`eviction_policy` 預設 lru [V1][V3]、要空間才淘汰 [V4]、從頭連續命中 [V9] | vLLM 是 block 層級、同一請求**尾端先趕** [V2]；session 引擎是整個 session 一起趕（X1 的語意） |
| E2 | CPU＋SSD；CPU 滿了把 LRU session 搬去 SSD（X1 的 `LRU`）；SSD 容量不限 | 04 §2 寫 LMCache；X6 的對應：Mooncake `offload_on_evict`（預設關）[M2]。**LMCache in-process 的預設其實是寫穿（H0）→ 比較像 E3** | SSD 容量／SSD 淘汰；水位式淘汰（Mooncake 90%/5% [M1]、LMCache MP 0.8/0.2 [L6]）——改成要空間才淘汰 |
| E3 | 每個寫進 CPU 的 block 同時寫一份到 SSD；CPU 淘汰只是丟 CPU 那份 | SGLang `write_through`（預設）[S1]；vLLM 0.28 tiering「Always offload to all tiers」[V6]；LMCache store（H0） | L1→L2 完成後才寫 L3（H0）；這裡和 CPU 寫入同一刻發出 |
| E4 | block 被用到第 2 次才寫 SSD；CPU 淘汰時沒有 SSD 副本的 block 就丟了（回來重算） | SGLang `write_through_selective`（命中 2 次）[S1]；Dynamo `presence_lfu`（`min_lfu_count`＝8）[D1]；AIBrix `HOT` [A1] | Dynamo 的門檻 8（04 §2 只跑 2） |
| E5 | E2＋session 閒置超過 T 就搬去 SSD；T＝300、1,800、3,600 s；閒置從該 session 上一個請求的到達時刻算（trace 時間） | Mooncake soft pin（預設 30 分，讀取不延長）[M1] | Mooncake 的語意其實是「soft pin 的最後才趕」；04 §2 把 E5 定義成「閒置超過 T 就搬」，照 04 做（CPU 在 T 之前就滿時，仍照 E2 的 LRU 搬） |
| E6 | E2＋請求一進佇列就開始從 SSD 讀它的 block（X4 的 QP）。**只有 serve=fifo1（一張 GPU、FIFO，X4 的事件引擎）才有佇列**；serve=trace（X1：照 trace 時間、不排隊）時沒有佇列，E6≡E2 | SGLang HiCache 進等待佇列就發 L3 預取 [S5]；LMCache async lookup 預取 [L1][L9] | SGLang 的預取 timeout（程式碼 1 s＋0.25 s／千 token [S6]；文件 2 s＋0.1 s／千 token、上限 30 s [S9]）：模擬器不讓請求先走，Cake 等在途的 block（X4 QP 的語意） |

---

## 3. 壓縮決定（`policies.py`）

- **我們只決定格式**。每個 session 在一輪結束、把**新 block** 寫進 CPU 時，引擎呼叫一次 `cpu_fid(ctx)`；
  外層把 block 搬去 SSD（E2／E5／E6 的降級、E3 的寫穿、E4 的第 2 次才寫）時呼叫 `ssd_fid(ctx, cur)`，回傳必須 ≥ cur（只能往下壓、不能變回原版）。
- 格式 id：1＝bf16、2＝8bit（Llama＝FP8、Qwen＝INT8）、3＝int4（KIVI）、4＝int2（KIVI）；id 越大越省、品質越差。
- 已經在 CPU 的 block（例如別的 session 寫的共用前綴）不重寫，保留原格式。
- **品質的「來歷」**：每份副本記它內容的最差格式（`cq`／`sq`）。從 SSD 的 INT4 讀回來再寫進 CPU，就算存成 BF16，品質仍記 INT4；
  只有重算（沒有任何副本、或在前綴的洞之後）才回到 bf16。這個記法是保守的：被 Cake 重算掉的前段，副本的來歷不會因此變好。

| 代號 | `cpu_fid` | `ssd_fid` |
|:--|:--|:--|
| C0 | bf16 | 不變 |
| C1（論文） | bf16 | int4 |
| C2 | int4 | 不變 |
| C3 | 8bit | 不變（Qwen 的 int8 在 placeholder 裡沒有大小 → 這格 NOT_MEASURED） |
| C4 | 寫入時的長度 ≤ 門檻 → bf16，否則 int4；**門檻是參數**（預設 32,768 token＝品質實測的長度），要在登記時凍結 | 不變 |
| O | 之後會被讀（或落在截尾區）→ 品質過關、位元組最少的格式；不會再被讀 → 位元組最少的格式（含 never_read_only 的 int2；沒有大小的跳過） | 不變 |
| 預測器 | `PredictorPolicy(fn)`：`fn(state, feats)` → 格式名或 fid | 可另給 `ssd_fn` |

- O 的資訊（只有 `needs_oracle=True` 的策略拿得到）：`next_read`＝之後第一次讀到這一輪**新 block** 的請求（任何 session，含分叉），
  `censored`＝決定時刻離觀察窗結尾不到 H 秒（「之後有沒有被讀」不算分 → 一律當會被讀）。
- O 的「最省」＝位元組最少，因為主指標是位元組（04 §4）。04 §3 說 O 也知道「從哪一層讀、那時 GPU 忙不忙」：這只影響 TTFT（報告指標），這版 O 沒拿來決定。
- O 的 `allow_unverified`（預設 False）：「未驗證」（例如 32K 以上的 INT4）不算過關。**這是登記要決定的事**（§8）。
- 預測器拿到的 `state` 去掉了 `next_read`、`censored`；`feats` 只放 trace 裡不含未來資訊的欄位。這一步不訓練任何模型（CLAUDE.md §8）。

---

## 4. 記帳（`ledger.py`、引擎內的計數）

| 帳 | 定義 |
|:--|:--|
| 每層每 1 秒窗口的讀寫位元組 | 每筆傳輸記 (開始, 結束, 位元組, 種類)；位元組依它在 I/O 線上**實際佔用**的 [開始, 結束) 平均攤到各秒（線上排隊的延後算在實際傳輸那一秒）。窗口從第一個請求到最後一筆傳輸結束，**含 0 的窗口**。層＝CPU（讀＝H2D 載入、寫＝存進 CPU）、SSD（讀＝還原與預取、寫＝降級／寫穿／第 2 次才寫／閒置降級） |
| 尖峰 | 所有窗口的 p99 與最大值（GiB/s），讀、寫、讀＋寫分開；**預取的讀取也算**，另給「只算預取」的一欄 |
| 總讀寫量 | I/O 線上的位元組（`Chan.bytes_r／bytes_w`）；另有依原因分的整數計數（store、demote、write_through、selective、ttl_demote；block 引擎：store、promote、demote、prefetch_read） |
| 品質帳 | 每個「有載入任何 block」的請求記一次讀取。這次讀取的格式＝載入的 block 裡**最差的品質來歷**；長度＝存的前綴 token 數（分箱＝格式表的 `len_bins_tokens`）；狀態照格式表（pass／fail／unverified）。**fail 的次數＝「品質沒過關的讀取」**；unverified 另列。回來的請求另外計 |
| 決定的分布 | 每次寫新 block 的格式 × 長度分箱 ×（O 才有）之後被讀？× 截尾？ |
| TTFT | 回來的請求（kind＝1）：Cake 還原（或 load-only）＋新一輪 prefill；平均、p50、p90、p99、還原的平均。fifo1 另含佇列等待（平均、p99）與 GPU 忙碌比例。**沒有 f(i)（70B）或解壓常數 NOT_MEASURED 的時間設定，TTFT 欄寫 NOT_MEASURED**（位元組照算） |
| 截尾版 | H>0 時另給：只算 [t0, 觀察窗結尾−H) 的窗口與這之前到達的回來請求 |

**記帳恆等式**（每格都檢查，冒煙與測試共用，`ledger.accounting_identities`）：SSD 寫入＝各原因加總＝Σ(寫入 block 數×該格式大小)；
CPU 寫入＝store（block 引擎：store＋promote）＝Σ(block 數×大小)；逐筆紀錄的加總＝I/O 線的總量（讀、寫、兩層）；block 引擎的預取讀取紀錄＝prefetch_read。

---

## 5. 場景軸

| 軸 | 怎麼做 | 參數在哪 |
|:--|:--|:--|
| GPU 忙閒 | f(i) × GPU 忙倍數（MI300X 2.07、3090 1.906）；解壓用忙時的常數 | View 的 `gpu` |
| CPU 忙閒 | CPU 層的讀（H2D）頻寬 ÷ 係數。**係數 NOT_MEASURED → 指定 cpu=busy 會直接丟錯**；3090 量到之後填進 `costs_<平台>.json` 的 `cpu_busy_h2d_factor` | View 的 `cpu_factor` |
| 模型的 KV 大小 | block＝kv_bytes_per_token×512：Llama 128 KiB（64 MiB／block）、Qwen 56 KiB（28 MiB）、70B 級 320 KiB（160 MiB）。70B 沒有 f(i)：只能 load-only、TTFT 不算 | 格式表 `models` |
| 成本常數 | `costs_mi300x.json`、`costs_3090.json`（§1.1）；SSD 另選 `ssd_profiles` | `load_costs(platform, model, ssd_profile)` |
| 幾張 GPU 共用一顆 SSD | SSD 讀、寫頻寬 ÷ 張數（04 §5） | View 的 `ssd_share` |
| 截尾修正 | 決定時刻離（這一列的）觀察窗結尾不到 H 秒：O 不准當「不會再被讀」；記帳另給截尾版（§4） | `H` |
| 負載疊加 ×2、×4 | copy c≥1 的每個 session 依 seed 抽一個位移，整個 session 在觀察窗內循環錯開（session 內的間隔不變），block 換一個命名空間（copy 之間不共用前綴）；`meta.tier='H'`，輸出標**半合成** | `trace.prepare(overlay=k, seed=…)` |
| serve=fifo1 的負載 | 一張 GPU 撐不住整份 trace（例如 Mooncake 整份的 GPU 忙碌比例 >1）時，以 session 為單位依 seed 抽樣（X4 的子集做法） | `trace.prepare(session_frac=p, session_seed=…)` |

### 5.1 資料（`trace.py`）

- Mooncake（512-token）：完整 block＝`block_ids[:input//512]`；Bailian（16-token，ceil）、FreeInference（16-token，floor）：每 32 個併成 512-token。
- block 的 key＝**整段前綴**（滾動：上一個 block 的 key＋這 512 token 的 id）。原始 id 是鏈式的來源（coder、thinking、FreeInference、Mooncake）用快的寫法「第 32j+31 個 id」（x1_fi），
  兩者一一對應（測試確認 P 完全相同）；**Bailian traceA 的 id 不是鏈式**（§10-2），只能用整段前綴。
- **規則 6 的斷言**（寫在 `prepare` 裡，不過就丟 `TraceAssertion`）：`len(block_ids)` 對 `input_tokens` 的 floor 或 ceil 吻合 ≥0.95；同一個 key 永遠在同一個位置（違反 0 次）；
  父請求不晚於子請求（子請求比父早開始的，夾到父的到達時刻並計數，比例 >1e-4 才丟錯；§10-3）。原始 id 的鏈違反數只報告。
  實測（冒煙的小窗）：Mooncake strict ceil 1.0、Bailian coder ceil 1.0、traceA ceil 1.0、FreeInference floor 1.0；key 的位置違反都是 0；原始 id 的鏈違反：traceA 前 30 分 40,554，其他 0。
- 沒有 block 的來源（WildChat、synth）：每個 session 自己一條鏈，**session 之間不共用前綴**（近似）。
- 「回來的請求」＝有父請求的那一輪（統一表的 `parent_turn_idx`）；切窗切掉父的，當成新的。
- 重播是**開迴路**（照 `t_start_ms`）：03_data_design §5 的預設是閉迴路，這版沒做（§8）。

---

## 6. 不變量測試（`test_p3_sim.py`）

**測試不讀 placeholder 格式表**：每個測試用 `_test_format_table()` 在暫存目錄造一份（bf16 64 MiB、8bit 33 MiB、int4 20 MiB、int2 12 MiB；
8bit 每個長度都過、int4 <16K 過 ≥16K 不過、int2 只給不會再被讀的）——**這些數值只為了測程式，和真實無關**。成本常數用 `costs_mi300x.json`（真的常數）。
結果 CSV 寫在測試 run 的目錄。

| # | 內容 | 結果（run `20261018-032844-p3sim-test（28 個測試全過）`） |
|:--|:--|:--|
| (i) | C1＋E2（X1 原樣）、C0＋E2（X1 的 INT4 大小換成 BF16、解壓 0）、C1＋E1（X1 的 `LRU+NONE`），在 X1 的設定上（`x1_sim.prep`：conversation R1 128／1024 GiB、R2 256 GiB、toolagent R1 128 GiB）× X1 的 12 種時間設定，比逐請求 TTFT、還原時間、CPU／SSD 載入數、重算數、SSD 讀寫位元組、CPU 寫入位元組 | **144／144 逐位元相同** |
| (ii) | block 引擎 E2／E6 vs `x4_sim.Sim`（NP／QP，一張 GPU FIFO）在 X4 的子集上（conversation p＝0.2／0.3／0.4、toolagent p＝0.2；32–128 GiB；local／nfs／b32），比逐請求 TTFT、佇列等待、qp_reads、demote、qp_stop_guard、pf_used、pf_evicted、promoted | **8／8 逐位元相同**；另外 serve=trace 時 E6≡E2（逐位元） |
| (iii) | O 對 C0–C4：Mooncake strict（前 30 分）、Bailian coder（前 30 分）、synth（前 4 小時）× E2／E3／E4／E5(300 s) × 16／64 GiB × H＝0／600 s；另 block 引擎 E2／E6（Bailian coder 30 分、session 抽 0.3） | O 的品質帳（fail）**全部 0**；SSD 讀寫總量 O ≤ 每個品質帳為 0 的 C：**120／120**（block 引擎 10／10）。**SSD 尖峰 p99：118／120，2 格違反**（synth、E2 與 E5、16 GiB，對 C3：O 6.691947 vs C3 6.691897 GiB/s）——測試只對總量下斷言，p99 照錄（§10） |
| (iv) | O 的容量單調：同上三份資料 × E2／E3／E4／E5 × 4–256 GiB（7 個） | SSD 讀寫總量不增：**84／84** |
| (v) | 同設定同 seed（Bailian coder 20 分、疊加 ×2、session 抽 0.5、E5＋O、fifo1 E6＋C4）跑兩次 → 輸出列＋逐請求陣列＋逐筆 I/O 紀錄的 sha256 相同；換 seed 不同 | **相同／不同** |
| 其他 | 品質來歷（C1：SSD 副本都是 int4；「全部從 CPU 讀、但來歷是 int4」的讀取存在）、截尾（H 大於觀察窗時 O 不選 int2，H＝0 時會選）、預測器拿不到 `next_read／censored／max_read_tokens` 且「永遠回 bf16」的預測器逐位元等於 C0、和 X8 的成本模型對拍（`x8_sim.cake` vs `View.restore`，cpu／local／nfs × 8 個長度：最大差 0.0 s）、NOT_MEASURED 會丟錯（3090 CPU 忙、/ssd8、沒有大小的格式）、鏈式來源的兩種 block key 給出相同的 P | 全過 |

### 6.1 和 X1／X4／X8 對拍

- **X1**：見 (i)。session 引擎的 restore／_fix／demote／enforce 是照抄 X1 的算式，格式推廣後在 C0／C1 退化回 X1，逐位元相同。
- **X4**：見 (ii)。block 引擎的 serve／cake／after_round／_evict_one／queue_prefetch（增量版）照抄 X4。
- **X8**：X8 是 session 層級、GPU 釘住＋TTL 的模型，和這裡的外層做法不是同一個機制（X8 的 TTL 管 GPU→CPU，E5 管 CPU→SSD），**沒有共同的策略設定可以逐位元比**；
  能比的是成本模型：單層均一 layout 的 Cake 還原時間＝`x8_sim.cake`（|差| 0.0 s，X8 自檢的門檻 1e-9）。E5 的 TTL 機制（到期 heap、版本號）的寫法參考 X8，但沒有對拍。

---

## 7. 小樣本冒煙（`smoke.py`，run `20261018-030356-p3sim-smoke`）

- 每份資料一小段：FreeInference week01（整週，10,738 個請求、125 個 session）、Bailian coder／traceA（前 30 分）、WildChat（第 200 天起 6 小時，969 個請求）、
  Mooncake conversation strict（前 30 分，5,719 個請求）、synth small-seed1（前 4 小時）。
- 每份 × E1、E2、E3、E4、E5（T＝300／1,800／3,600）、E6（block 引擎 fifo1，另跑同引擎的 E2）× C0–C4、O；CPU 64 GiB；H＝1,800 s；
  時間設定：MI300X 閒、MI300X 忙＋SSD 4 張共用、MI300X 閒 load-only、3090 閒（fifo1 只有第一個）。
- 另外各一格：Qwen（MI300X 閒、3090 忙）、70B 級（load-only）、疊加 ×2／×4（Bailian coder，**半合成**）、Mooncake 抽 0.25 的 session 跑 fifo1。
- **格式表是 placeholder**（sha256 `58b798d0…6c84`）→ 品質一律 unverified，品質帳只確認有在記；**這次冒煙沒有算、也沒有看任何「省多少」**。
- 結果：1,096 格、記帳恆等式 7,746 條**全部成立**（§4 的每一條，session 引擎 1,022 格、block 引擎 74 格）；E1 的 SSD 讀寫全部是 0（只有 CPU 層）；
  3090 的 Llama 時間設定 TTFT＝NOT_MEASURED（解壓常數沒有），位元組照算。
- 錯誤 2 條，都是預期中的 NOT_MEASURED：70B 的 C1、C2（int4 沒有大小）。
- fifo1 的 GPU 忙碌比例（一張 GPU 撐不撐得住）：Mooncake 整段 3.79、Bailian coder 1.27（>1＝撐不住，佇列一直長）；Mooncake 抽 0.25 → 0.96；
  traceA 0.56、synth 0.24、FreeInference 0.05、WildChat 0.002。**E6 的格子在完整地圖要先決定每張 GPU 分多少 session**（§8）。
- 第一次冒煙（run `20261018-024805-p3sim-smoke`）在 O 的修正後被我中止（目錄裡有 `ABORTED`）；第二次（`20261018-025459-p3sim-smoke`）抓到 Bailian traceA 的 block id 問題（§10），修正後重跑＝本節。

---

## 8. 已知限制（先講）

1. **開迴路重播**：照 trace 的到達時間。03_data_design §5 說預設閉迴路（下一輪＝模擬的完成時間＋think_ms），這版沒做；X1／X4 也是開迴路（逐位元重現需要）。
2. **session 引擎是整個 session 一起趕**（X1 的語意），block 引擎是 block 層級 LRU、同時間先丟前面（X4／m7 S4L）。**兩者都不是真實系統的「尾端先趕」**（vLLM、LMCache 反序 touch，X6）。還原都只用最長的有存前綴。
3. **E6 只有 fifo1（一張 GPU、FIFO、不批次、沒有 chunked prefill）**；serve=trace 時沒有佇列（E6≡E2）。E6 的 capture（GPU→CPU）用 CPU 層的寫入速度（X4 用 C0 的 D2H 44.8 GiB/s）；逐位元對 X4 的測試是用 X4 的值。SGLang 的預取 timeout 沒做。
4. **SSD 容量不限、不淘汰**；水位式淘汰沒做（§2）。
5. 存的 KV＝這一輪輸入的完整 512-token block（X1 的定義）；輸出的 KV 要到下一輪的輸入才存。
6. **格式決定**：每個 (session, 一輪) 對這一輪新寫進 CPU 的 block 決定一次；別的 session 寫的共用前綴保留原格式。
7. **品質來歷是保守的**：被 Cake 重算掉的前段，副本的來歷不會變好；一份讀取的格式＝載入的 block 裡最差的。
8. **O 是貪婪的參考線，不是上界**：目標是位元組最少；「之後每次讀都要過關」用之後讀到這些 block 的請求裡最長的存的前綴（讀取長度的上界）；沒有用讀取時的層與 GPU 忙閒。(iii)、(iv) 只在小樣本上驗。SSD 尖峰 p99 有 2／120 格 O 比品質帳 0 的 C3 高（差 5e−5 GiB/s）。
9. E5 的閒置從 session 上一個請求的**到達**時刻算（不是回應結束）；E4 的「被用到 2 次」＝含這個 block 的請求數到 2。
10. **placeholder 格式表**：品質全 unverified、int2 與 Qwen int8 沒有大小。正式表的結構不同，要寫轉換（§1.2）。O 的 `allow_unverified` 預設 False。
11. WildChat、synth 沒有 block id：session 之間不共用前綴。synth 的 token 是 o200k_base 直接當 512 token。
12. 70B 級沒有 f(i)：只能 load-only、TTFT 不算。CPU 忙係數、3090 /ssd8、3090 的 Llama 解壓：NOT_MEASURED。
13. 1 秒窗口：每一輪的 SSD 寫入在時間軸上合成一筆（X1 的寫法；逐位元需要），所以同一輪的多個 block 是一筆連續傳輸。
14. 多個時間設定共用一次放置：只有放置和時間無關時才成立（E1–E5、serve=trace），fifo1 一次只能一個時間設定。
15. block 引擎的 O 每次決定掃一次之後的讀取（沒有預先算後綴最大值）；完整地圖前要改成和 session 引擎一樣的後綴最大值（目前只影響速度）。

### 8.1 要在登記時決定的事（模擬器都做成參數）

C4 的長度門檻（預設 32,768）；O 能不能用「未驗證」的格式；開迴路或閉迴路；E6 每張 GPU 分多少 session（`session_frac`）；
E2 代表誰（04 寫 LMCache，但 LMCache 的預設是寫穿 → E3）；session 引擎或 block 引擎當主地圖；E1 的主指標（E1 沒有 SSD，SSD 位元組恆為 0，「省多少」沒有定義，只能看 TTFT／重算）。

---

## 9. 完整地圖要跑多久〔估計〕

量的（run `20261018-032839-p3sim-timing`，`timing.csv`；共用機、同時有別人的負載）：一格＝E2＋C1；session 引擎一次 8 個時間設定（GPU 閒／忙 × SSD 共用 1／2／4／8），block 引擎（E6）1 個。

| 資料（整份） | 請求數 | 每請求平均 block | 準備（秒） | session 引擎 8 views（秒，32／512 GiB） | block 引擎 1 view（秒） | fifo1 GPU 忙碌比例 |
|:--|--:|--:|--:|:--|:--|--:|
| Bailian coder | 43,011 | 10.7 | 7.2 | 3.8／3.9 | 4.0／9.2 | 1.19 |
| Bailian traceA | 43,058 | 4.1 | 4.9 | 2.0／1.9 | 1.7／1.4 | 0.70 |
| Mooncake conv strict | 12,031 | 23.0 | 1.0 | 1.3／1.4 | 2.3／3.3 | 3.27 |
| FreeInference week13 | 163,970 | 385.2 | 561.7 | 286／376 | 627／140 | 0.87／0.63 |
| WildChat（7 天窗） | 32,199 | 0.78 | 0.6 | 3.7／1.6 | 0.6／0.5 | 0.002 |

**估計**〔估計：用上表的每請求秒數線性外插；FreeInference 16 週共 1,222,199 個請求，假設每請求的成本和 week13 相同；WildChat 整份 1,960,074 輪〕：
- 主地圖的 session 引擎部分（04 §5：5 份資料 × E1、E2、E3、E4、E5×3 × CPU 5 種 × C0–C4＋O×2 種 H＝245 次放置，每次 8 個時間設定）：
  每次放置 ≈ FreeInference 2,100–2,800 s＋WildChat 100–220 s＋其他 ≈ 7 s → 約 **160–210 CPU 小時，8 個 process ≈ 20–26 小時**。
- E6（block 引擎，E2／E6 × CPU 5 × 7 種壓縮 × 8 個時間設定＝每份資料 560 次）：FreeInference 每次 1,000–4,700 s → **160–720 CPU 小時（8 個 process 20–90 小時）**；其他資料合計 < 10 CPU 小時。
- 準備：FreeInference 16 週 ≈ 70 分（每個 process 一次；建議存成快取）。記憶體：FreeInference week13、8 個時間設定的 session 引擎約 10.8 GB／process。
- **結論〔判讀〕**：時間幾乎都花在 FreeInference（每請求 385 個 block）。跑完整地圖前建議先做兩件不改語意的事：(1) block 引擎的 O 改成後綴最大值、after_round／serve 的逐 block 迴圈減量（§8-15）；(2) 準備好的 P 存成快取。做完要重跑 (i)、(ii) 確認仍逐位元。

---

## 10. 失敗與異常（完整錯誤）

1. **測試 (iii) 第一次沒過**（開發中，scratch 目錄、沒有走 run3090.sh、沒有 run_id）：
   ```
   E           AssertionError: O 的品質帳不是 0：E2 16 1408
   FAILED code/p3_sim/test_p3_sim.py::test_iii_oracle_dominates_quality_ok_fixed[0.0-mooncake-conv-rule-strict-0-1800-kw0]
   FAILED code/p3_sim/test_p3_sim.py::test_iii_oracle_dominates_quality_ok_fixed[0.0-synth-small-seed1-0-14400-kw2]
   FAILED code/p3_sim/test_p3_sim.py::test_iii_oracle_dominates_quality_ok_fixed[600.0-mooncake-conv-rule-strict-0-1800-kw0]
   FAILED code/p3_sim/test_p3_sim.py::test_iii_oracle_dominates_quality_ok_fixed[600.0-synth-small-seed1-0-14400-kw2]
   FAILED code/p3_sim/test_p3_sim.py::test_iii_fifo1_oracle - assert 273 == 0
   ```
   原因（O 的定義錯了，不是測試錯）：O 只看**寫入當下**的長度選格式；同一批 block 之後會被**更長的**請求讀到（session 長大），而格式只能往下壓，
   所以短的時候壓成 int4、之後在 ≥16K 讀到就不過關。修正：O 用「之後讀到這些 block 的請求裡最長的存的前綴」選格式（`max_read_tokens`）。
   block 引擎另有一個錯：沒有新 block、只有 promote 回 CPU 的一輪，被當成「不會再被讀」→ int2；修正：看這一輪實際要寫進 CPU 的第一個 block。修正後 (iii) 全過。
2. **冒煙抓到 Bailian traceA 的 block id 不是鏈式**（run `20261018-025459-p3sim-smoke`，`smoke_errors.csv`）：
   ```
   ERROR bailian_traceA prepare TraceAssertion('bailian/traceA: 前綴鏈被違反 38 次（同一個 block key 出現在不同位置）')
   ```
   查原因（統一表 `bailian/traceA` 的原始 16-token id）：**52,168 個 id 出現在不只一個位置**（位置違反 220,196 次、前一個 id 違反 81,514 次）；coder、thinking 都是 0。
   所以 x1_fi 的「第 32j+31 個 id 代表整段前綴」對 traceA 不成立（不同前綴會被當成同一個 block）。
   修正：block 的 key 改成**整段前綴**（滾動：上一個 block 的 key＋這 512 token 的 32 個 id）；原始 id 是鏈式的來源仍用快的寫法，
   測試確認兩種 key 在 coder／Mooncake／FreeInference 給出完全相同的 P。原始 id 的鏈違反數照錄在 `meta.checks.raw_id_chain_violations(report)`。
   **這是資料層的發現，要回報**：03b 的 Bailian 斷言有「子 hash 前綴＝父 hash[:-1]」，但沒有檢查跨請求的鏈性質；用 traceA 的 hash 當前綴 id 的程式都要注意。
3. **計時 run 抓到 FreeInference 的子請求比父請求早開始**（run `20261018-030356-p3sim-timing`，exit 1）：
   ```
   trace.TraceAssertion: freeinference/week13: 父請求排在子請求之後（472 ≥ 471）
   ```
   查原因：16 週裡只有 1 列（week13，session `b6016c383d98…` 的 step 7 比 step 6 早 14 ms 開始；03b §6.4 也記過 FreeInference 有 17 列 think_ms<0）。
   處理（**偏離，請主 session 確認**）：子請求的到達時刻夾到父請求的到達時刻、排在父後面，計數記在 `meta.checks.child_before_parent_clamped`
   （week13＝1），比例超過 1e-4 仍丟錯。修正後重跑（`20261018-032839-p3sim-timing`）。
4. 冒煙的 2 條 NOT_MEASURED（70B 的 C1、C2）是預期中的，不是錯。
5. 測試 (v) 的指紋含格式表的暫存路徑，所以只在同一次測試內可比（不同次的 run 指紋不同是正常的）。
