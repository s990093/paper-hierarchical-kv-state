# 13　還原的第二輪：GPU 忙時的「拆」、會合規則、新格式 kernel、拆進 attention、長 context（事先登記；只報告）

> 回到 [README](README.md)。2026-10-17 UTC 02:4x 寫。
> **使用者 10/17**：「反量化在我們問題上很重要，所有 SOTA 需要的都拿來測試跟實驗嗎？」「跟哪樣時機？哪樣配合？比較好」「（1–4）全部都做掉，目前感覺 GPU 忙時，『拆』要和別的工作搶 GPU 比較嚴重」「拆直接融合進 attention 感覺也可以試試看？類似對照組？」
> **只報告，不改主線判定**（07 的速度判定用原寫法、Cake）。要拿來改判準，必須老師同意、另外登記（08 §6 第 1 點選項 c）。

## 0. 寫這一份時已經知道的（全部是 32K）

- 42（融合 kernel，K1、S0）：真實 CPU 層的 Cake 還原，低精度不再比 BF16 慢（[11 問題一 §3a](11_issues.md)）。
- 42b 海選（INT4、CPU 層）：「8 塊一組一起拆」（S2k8）最好；GPU 忙時所有方法差不多（11 問題一 §3b）。
- 已有速度格子的整理（11 問題三）：CPU 層「全部載入」每一格都比 Cake 快 14–55%；全載入時 GPU 忙的「拆」佔還原時間約一半（Llama INT4 29／61 ms、Qwen INT4 23／42 ms）。
- 42c 複賽跑中（28 個 grid 跑完 16 個）。我只看過其中一格的數字（Llama FP8、S2k8、CPU 層：E1 閒 0.064 s、忙 0.085 s；寫分析程式時的測試）。
- **下面每一項都還沒有任何結果。**

## 1. 要回答的

| 代號 | 問題 | 使用者的優先 |
|:--|:--|:--|
| Q4 | GPU 忙時，「拆」和別的工作搶 GPU：哪些做法能讓它少被拖慢？代價是誰付？ | **最優先** |
| Q2 | 允許「完全不重算」的會合規則（G5），能不能在每一層都接近 Cake 和全載入裡比較快的那個？ | 一定要做 |
| Q3 | 大掃描的新格式（INT2、MXFP4、K／V 分開的格式）換成融合 kernel，CPU 層快多少？ | 做 |
| Q5 | 把「拆」直接融合進 attention（還原時不拆，用的時候才拆）：還原省多少、之後每一步多花多少？ | 對照組 |
| Q1 | 上面的做法在 64K／128K 還成立嗎？ | 一定要做 |

比喻（給讀者）：搬家用壓縮袋。Q4＝搬到的時候大家都在用客廳，拆封要排隊；Q2＝東西就在隔壁時，別自己重做一箱，直接全搬；Q3＝新形狀的壓縮袋也配一把好用的剪刀；Q5＝乾脆不拆，要用哪件才從袋子裡抽出來；Q1＝家當變成 2–4 倍時還一樣嗎。

## 2. 共同設定

- 模型 Llama、Qwen；格式：每個模型主線的 FP8（Llama fp8_token、Qwen fp8_head）和 int4_kivi（Q3 另見 §5）。
- 拆：融合 kernel，K\* 用 42b／42bz A 段選出的（`results/m10_sweep_mi300x/dqsweep/A/choice_<m>.json`），64K／128K 也用同一組 K\*。
- 背景負載：主線的 `m7_busy.py`（另一個 process 做 8192² 的 BF16 matmul，duty 0.7＝busy2）。
- 計時、污染：`m10_speed` 的 worker；每個 launch 開 gpu_guard，不是 CLEAN 的 run 時間不能用、要重量（CLAUDE.md §3）。每格第一次還原做 KV 逐位元檢查（`kv_bad`），沒過的方法淘汰並寫 NEEDS_HUMAN（Q5 例外，§6）。
- launch 編號從 300 起（和 42、42b、42c 不同）。
- 配對差、CI、「不輸」：同 `m10_dqsweep.py report`（配對＝同一個 launch、rep、頻寬、GPU 狀態；bootstrap 95% CI）。δ\* 用主線凍結的 noise.json（Llama 5%、Qwen 9.43%）。
- 新的會合規則代號（環境變數 `M10_MEET`）：`claim`＝Cake（原本）；`estimate`＝G4；**`none`＝全部載入，不重算**（走和 Cake 同一條載入線，所以 S 組的拆法都能用；和舊的 `load_only` 參考線不同，那條一定逐塊拆）；**`g5`**（§4）。

## 3. Q4：GPU 忙時的「拆」（最優先）

### 3.1 方法

| 代號 | 做法 | 依據 |
|:--|:--|:--|
| P0 | 融合 kernel（K\*），逐塊拆（S0） | 42 的做法，基準 |
| P1 | P0＋還原用的 stream（H2D、拆）設最高優先權 | GPU 的 stream 優先權；42b 的 S3 是讓「重算」優先，這裡是讓「拆」優先 |
| P2 | 8 塊一組一起拆（S2k8） | 42b 海選冠軍；kernel 數少 8 倍 |
| P3 | 全部搬完一次拆（S4） | 只開一個 kernel |
| P4a | P0＋背景負載限制在 7/8 的運算單元（ROCm `HSA_CU_MASK`），留 1/8 給還原 | 空間切分（AMD 才有的開關）；ShadowServe 講的「解壓和計算搶 GPU」 |
| P4b | 同上，背景限制在 3/4 | 留更多 |
| P5 | P1＋P2 | 兩個都便宜，看會不會疊加 |

- P4 的遮罩只加在背景負載的 process（對應真實系統裡「同一個引擎替其他請求保留／限制運算單元」）。CU 編號在 MI300X 的 8 個 XCD 上怎麼分布，冒煙時先確認：限制之後背景單獨跑的 matmul 時間有照比例變長，遮罩才算生效；沒生效就拿掉 P4、寫 NEEDS_HUMAN。
- **代價要一起報**：P1、P4 會讓背景工作變慢。`m7_busy.py` 結束時印出它做了幾次 matmul、跑了多久 → 每個 GPU 狀態的「背景吞吐（matmul／秒）」，和 P0 同一格比。

### 3.2 量什麼

- 真實 CPU 層；GPU 閒（對照）／忙；會合規則 `claim`（Cake）和 `none`（全載入）；配置 E0、E1（E0 也要量：優先權、遮罩也會改變 BF16）。
- 3 launch × 10 rep。每個方法、模型、格式一個 grid：7 × 2 × 2＝28 個 grid。
- 報告：每格 E1、E0 中位數；E1 對 E0 的配對差與 CI；拆的時間（`t_dequant`）；背景吞吐對 P0 的比值。
- **帶到長 context 的方法**（寫死）：在「背景吞吐 ≥ P0 的 95%」的方法裡，取「GPU 忙、全載入、E1 中位數」對 4 格（2 模型 × 2 格式）的幾何平均最小的；和 P0 差不到 3% 時取 P0。被遮罩的 P4 另外報，不參加挑選（它讓背景付錢）。

### 3.3 背景強度掃描

- P0 和上面挑出來的方法；int4_kivi、兩個模型；duty 0.4（busy1）、0.7（busy2）、0.9（新的，沒校準過，只當「更忙」的方向）；`claim` 和 `none`；E0、E1；2 launch × 5 rep。
- 報告：還原時間隨 duty 怎麼變（每格中位數）。

## 4. Q2：會合規則 G5（允許完全不重算）

- **規則**：開始還原前，用「這個（頻寬, GPU 狀態）之前幾次還原量到的」每塊載入時間（含拆，依存的格式分開）和第 0 塊的重算時間估計：
  - 所有 block 都有存，而且「Σ 每塊載入時間 ≤ 第 0 塊重算時間」→ 不重算（同 `none`）。
  - 否則照 Cake（`claim`）。
  - 還沒有紀錄時照 Cake。每個（頻寬, GPU 狀態）在計時前先做一次不記錄的 E0 和 E1 還原（Cake）當 profiler（Cake 自己也是事先量），之後每次還原都更新紀錄（中位數）。
  - 每列記下 G5 的決定（`meet_rule_used`）。
- **量什麼**：K\*、S0；會合規則 `claim`、`g5`，CPU 層另加 `none`；頻寬 CPU、b16、b32、b48；GPU 閒／忙；配置 E0、E1、R25。
  - CPU 層 3 launch × 10 rep；限速層 2 launch × 5 rep（b48 一次還原要 1.5–4 秒）。
- **看什麼**（描述，不是判準）：每格 G5 的中位數和 min(Cake, 全載入) 差多少；G5 在 CPU 層有沒有選全載入、在 b32／b48 有沒有照 Cake。b16 是最接近的一層，報 G5 選錯的比例。

## 5. Q3：新格式的融合 kernel

- 格式：大掃描複賽（43）用到的 int2_kivi、mxfp4、kv-int4_kivi-int2_kivi、kv-int8_token-int4_kivi（K、V 分開時各自用對應的 kernel）。
- 正確性：GPU 上和原寫法（`m10_codec_x` 的 decode）**逐位元相同**，不同就不用、寫 NEEDS_HUMAN。先在 CPU 直譯模式（Triton interpreter）測。
- 量：43 的速度配置（`queue/43_sweep_confirm.sh` 的 grid），只跑真實 CPU 層，`M10_DEQUANT_FAST=1`；3 launch × 10 rep；和 43 原寫法的 CPU 層並排。
- **43 的複賽判定照原寫法不動**（43 是同格式、同佔比之間比，用同一種拆法，公平）；這裡只報告 CPU 層的絕對時間。

## 6. Q5：把「拆」融合進 attention（對照組）

**是什麼**：現在是還原時就把每個 block 拆回 BF16，之後 attention 讀 BF16。融合進 attention（QServe、BitDecoding、PackKV 的做法）是還原時**只搬不拆**，attention 每次讀到壓縮的 KV 才在暫存器裡拆。還原變快，但新一輪的 prefill 和之後**每一步 decode** 都要多拆一次。所以它是「還原省下的」對「每一步多花的」的取捨，要算出「生成幾個 token 以內划算」。

它換了 attention 的寫法（和 E0 不是同一條 attention，累加順序不同，答案不會逐位元相同），違反 07，所以**只能當對照組**，不進任何判定。

- **(a) 只搬不拆的還原**：會合規則 `none`＋新拆法 `nodq`（壓縮的 bytes 搬進 GPU 暫存區就結束）。量還原時間；CPU 層、b32；閒／忙；E1；3 launch × 10 rep。這是「融合進 attention」在還原這一段的實際時間。
- **(b) attention kernel 微測**：自己寫一個 Triton attention（GQA、因果），三種讀法共用同一份程式碼：讀 BF16、讀 FP8（per-token／per-head scale）、讀 INT4 KIVI（讀的時候拆）。
  - 兩種形狀：新一輪的 prefill（256 個 query）和 decode（1 個 query）；context 32K、64K、128K；每個模型的 head 數、層數（量一層 × 層數）；GPU 閒／忙；每格暖機 5、量 50，中位數。
  - 另外量 PyTorch SDPA 讀 BF16（現在的路徑）當業界參考。
  - 數值：和「先拆成 BF16 再用 SDPA」比，報最大絕對誤差、相對誤差（不要求逐位元）。
- **(c) 組合**（〔算術〕，用 (a)(b) 的實測）：TTFT ≈ 只搬不拆的還原＋新一輪（實測的 t_new_turn − SDPA 那段＋讀壓縮的 attention）；每生成一個 token 多花 Δ＝(讀壓縮 − 讀 BF16)×層數；**划算的 token 數＝還原省下的時間 ÷ Δ**。
- **(d) 業界參考（有空才做）**：vLLM 0.28（ROCm）`--kv-cache-dtype fp8` 和 `auto` 的 decode 每 token 延遲（32K、128K），vLLM 的 FP8 KV 就是在 attention 裡拆。只報告。
- 沒有做的：把它接進整個模型的 forward、量答案品質。(b)(c) 的結果顯示值得，才另外登記。

## 7. Q1：長 context（64K、128K）

- 方法：P0（K\*、S0）；42c 的冠軍（**挑法寫死**：42c 的 5 個選手裡，CPU 層 E1 中位數對 8 格〔2 模型 × 2 格式 × 閒／忙〕的幾何平均最小的；是 S0 就只跑 P0）；Q4 帶過來的方法；G5；只搬不拆（Q5 (a)）。
- 會合規則：`claim` 和 `none`（G5 另外跑）；頻寬 CPU、b32；GPU 閒／忙；配置 E0、E1、R25；2 launch × 5 rep（同 44a）。
- 和 44a（原寫法、Cake＋只載參考線、所有層）並排。
- 報告：每格中位數、E1 對 E0 的配對差、拆的時間、重算幾塊。

## 8. 順序、時間、上限

- 佇列（使用者 10/15：主線、掃描在前；10/16：長 context 在只報告的工作前；10/17：這一輪全部做，Q4 優先）：
  `43 大掃描複賽` → **43a 冒煙**（新程式在 GPU 上：逐位元、遮罩生效、G5／none／nodq 各一次）→ **43b Q4** → **43c Q2** → **43d Q3** → **43e Q5 (a)(b)** → 44 → 44a → 44b → 44c → **44d Q1** → 45 → 46 → 55 → 60 → 70 → 75 → 88。
- 新程式的 GPU 檢查只能排進佇列（同一張卡，跑的時候會污染別人的計時）；CPU 上能測的先測完。
- 〔估計〕GPU 時間：43a 0.3 小時；43b 1.5＋0.7 小時；43c 1.5 小時；43d 0.7 小時；43e 1 小時；44d 2.5 小時；合計約 8–9 小時。全部跑完從 10/20 晚上延到約 10/21，還在 10/23 前。
- 上限：這一輪合計 12 小時。快超過時照這個順序砍：先砍 Q4 的 duty 0.9，再砍 Q2 的 b48，再砍 Q1 的 64K。
- 冒煙沒過：只把用到那段程式的 stage 移到 `hold/`、寫 NEEDS_HUMAN，其他照跑。

## 8a. 補充（10/17 03:1xZ，寫程式時定的細節；仍然沒有任何 13 號的結果）

- **Q4 的 claim／loadall**：`claim` 列＝Cake，用各方法自己的拆法（P2 是 8 塊一組、P3 是全部搬完一次拆）；`loadall` 列＝全部載入，也用各方法自己的拆法。兩種在同一個 process 裡、同一次重複內交錯（m10_speed `--extra-modes loadall`）。
- **P4 的遮罩**：43a 冒煙用 `m10_restore2.py cumask` 算：CU 數 n（MI300X 預期 304）→ P4a `HSA_CU_MASK=0:0-(7n/8−1)`、P4b `0:0-(3n/4−1)`。生效＝背景單獨跑的 matmul 時間 ≥ 沒遮罩的 ×1.05（P4a）／×1.10（P4b）；沒生效 → 43b 不跑 P4。
- **Q2**：G5 在同一個 process 裡，每個配置另外跑 `claim`（Cake），CPU 層再加 `loadall` → 每一列都和同一次重複的 Cake、全載入配對。G5 的 profiler：每個 GPU 狀態開始前，每個頻寬照 Cake 跑 E0、E1 各一次，不記錄。
- **Q3**：新格式沒有調參，一律 k1；43 速度配置裡的 int4_kivi 也用 k1（同一個 grid 一致）。launch 0–2（和 43 同 seed，R 的位置相同）。
- **Q5 (a)**：配置 E0、E1；CPU 層、b32。
- **Q1（44d）的範圍**（照 §8 的上限先縮，避免 b32 的全載入在 128K 一次要約 11 秒）：
  - P0、G5：CPU 層＋b32。冠軍、Q4 帶過來的方法、nodq：只跑 CPU 層。
  - `loadall` 只在 CPU 層（b32 在 32K 已經知道全載入慢 1.2–2 倍；128K 的 b32 全載入由 44a 的參考線量，原寫法）。
  - G5 的 process 另跑 CPU 層的 `claim`、`loadall`。
  - 〔估計〕約 2.2 小時。
- **43a 冒煙**：新格式 kernel 和 attention 對照組在 GPU 上自測（逐位元）；cumask；真模型 Qwen int4_kivi 1 launch × 1 rep：G5＋claim＋loadall（CPU、b32）、nodq、P5＋遮罩（GPU 忙）；Llama int2_kivi 走大掃描速度路徑＋新格式 kernel。每格第一次都檢查 KV。冒煙的時間不是結果。
- 佇列實際檔名：`43a_rs2_smoke`、`43b_rs2_busy`、`43c_rs2_g5`、`43d_rs2_newfmt`、`43e_rs2_attn`、`44cz_rs2_plan44d`（CPU：pick-c、寫 44d）、`44d_rs_longctx`（由 44cz 寫）。

## 8b. 補充（10/17 03:2xZ；使用者問「有沒有現成程式碼可以直接用」；仍然沒有任何 13 號的結果）

- 查的結果（摘要級，self-review）：AMD MI300X 上能直接用的「拆在 attention 裡」只有 **FP8**（vLLM 的 ROCm 路徑：AITER／Triton，`--kv-cache-dtype fp8`）。INT4 KIVI 的現成實作（KIVI 原作、QServe、BitDecoding、LMDeploy、FlashInfer）都是 CUDA，只能在 NVIDIA 上跑。
- 所以 Q5 加兩個外部參考（都只報告、只看時間）：
  - **§6 (d) 改成正式要跑**：`code/m10_vllm_kvdecode.py`，vLLM 0.28 ROCm，`auto`（BF16）vs `fp8`，每個 token 的 decode（(t65 − t1) ÷ 64）、32K 與 128K、每格 5 次。vLLM 的 FP8 是 MI300X 的 fnuz 版、自己的 scale，不是我們的 codec。Qwen 1M 模型在 vLLM 0.28 的 128K 可能起不來（DCA），起不來就照實記。
  - **vLLM-KIVI 的 Triton kernel**（[github.com/LLAA178/vllm-kivi](https://github.com/LLAA178/vllm-kivi)，commit cd0f8e9，Apache 2.0；原樣抄在 `code/third_party/vllm_kivi_attention.py`）：`m10_attn_q.py bench` 的 decode（1 個 query）另加 `ext_vllm_kivi`。只量時間：它的 4-bit 格式和我們的不同（K 每 16 個 token 一組、V 每 token 一個 scale）、每個 q head 各讀一份 KV（不共用 GQA）、沒有 split-K、為 A100 調；它的 K scale 用 k_cache 的 stride 索引到 d＜128（看起來會碰到下一個 head 的位置），所以配一塊夠大的平面緩衝讓讀取都在範圍內，值是亂數。量一層 × 層數次。
- 這兩個都不換掉我們自己的 kernel：只有同一份程式碼讀 BF16 和讀壓縮，才能把「拆」的成本單獨量出來。

## 8c. 補充（10/17 04:1xZ；**加 G5b**；正式的 13 號結果仍然一個都沒有）

- **為什麼加**（照實寫）：43a 冒煙（Qwen int4_kivi、CPU 層、閒、1 launch × 1 rep；冒煙的時間不是結果）看到 G5 的 E0、R25 都選了 Cake，但同一次的全載入比較快（E0：G5 0.045、Cake 0.053、全載入 0.039 s）。原因在規則本身：G5 只拿「全部載完」和「重算第 0 塊」比，可是 Cake 在載入線還沒做完時會多開一塊重算（整數跳動，11 問題一 §2），所以 Cake 的實際時間比「重算 1 塊」長，G5 會低估全載入的好處。
- **G5b**（`M10_MEET=g5b`）：紀錄、profiler 和 G5 完全一樣；決定改成用同一組估計（每塊載入時間依來源、每塊重算時間＝第 0 塊的中位數）**模擬 Cake 的兩條線**（重算線從前面一塊一塊拿；載入線從後面拿；誰先空下來誰拿下一塊；到兩邊相遇），「全部載完 ≤ 模擬的 Cake」就不重算。CPU 測試：`tests/restore2_cpu_shim.py`（含「載入 3×8、重算 14」這種 G5 選 Cake、G5b 選全載入的例子）ALL PASS。
- **G5 照原登記照跑**，不拿掉、不改。43c 先用真模型冒煙 G5b（1 × 1，KV 逐位元、每列有決定），過了才跑 G5b 的 8 個 grid（和 G5 同樣的設定，目錄加 `_g5b`）。〔估計〕多約 50 分鐘。
- **長 context 帶哪一個**（`m10_restore2.py pick-g5`，寫死）：兩者都有可用資料的共同格子裡，各自「中位數 ÷ 最好」的幾何平均（最好＝CPU 層 min(Cake, 全載入)、限速層 Cake）；G5b 小於 G5 的 99% 才換成 G5b，否則用 G5。

## 9. 不做的

| 項目 | 為什麼 |
|:--|:--|
| CPU 上先拆再搬 BF16 | CPU 層要搬的 bytes 回到 BF16，省的全沒了 |
| 解壓搬到網卡、影片解碼器（ShadowServe、KVFetcher） | 這台沒有那種硬體 |
| AMD 硬體 FP8 轉換指令（Petit） | 融合後一塊只要約 0.08 ms，64 塊合計幾 ms；最多省幾 ms。Q4 的結果顯示拆本身還是瓶頸，再補 |
| 熵編碼（CacheGen） | 只在慢的層可能划算；要寫 GPU 解碼器，這一輪不做 |

## 10. 要寫的程式

- `code/m7_restore_harness.py`：會合規則 `none`、`g5`（含紀錄與 profiler）；拆法 `nodq`；還原 stream 優先權。預設路徑不變（逐位元、逐步驟相同）。
- `code/m10_speed.py`：環境變數 `M10_MEET=none|g5`、`M10_DQ_SCHED=nodq`、`M10_RS_PRIO=1`、`M10_BUSY_CU_MASK`、`M10_BUSY_DUTY`；CSV 加 `meet_rule_used`；背景吞吐寫進 run 目錄。
- `code/m7_busy.py`：結束時印 matmul 次數和時間（預設行為不變）。
- `code/m10_dequant_fast.py`：int2_kivi、mxfp4 的融合 kernel（K／V 分開的格式組合現成的）。
- `code/m10_attn_q.py`：Q5 (b) 的 Triton attention 與微測。
- `code/m10_restore2.py`：產生 43b–43e、44d 的佇列腳本，和這一輪的報告（CPU）。

## 11. H200 上要用的（使用者 10/17：「記下來 h200 上要用這些」）

NVIDIA H200（Hopper）上可以直接跑、MI300X 上不能跑的現成實作。H200 做這一輪時要拿來比（只報告；先照 [H200_SLURM](../H200_SLURM.md) 的規則：gpu_guard 在 Slurm 上是 UNOBSERVABLE_HOST、一個 job 最多 48 小時）：

| 實作 | 用來比什麼 | 為什麼 H200 才行 |
|:--|:--|:--|
| KIVI 原作（[github.com/jy-yuan/KIVI](https://github.com/jy-yuan/KIVI)，CUDA 的融合反量化矩陣乘） | 和我們的 int4_kivi／int2_kivi 最接近的「拆在 attention 裡」 | CUDA |
| BitDecoding（arXiv 2503.18773，Tensor Core 低位元 KV decode） | 低位元 KV decode 的 SOTA 速度 | CUDA、NVIDIA 的 mma 指令 |
| QServe（arXiv 2405.04532，W4A8KV4） | INT4 KV 的融合 attention | CUDA |
| FlashInfer（FP8 KV 的 prefill／decode attention） | FP8 讀進 attention | CUDA |
| vLLM `--kv-cache-dtype fp8`（Hopper：OCP e4m3，硬體轉換） | 和我們的 codec 同一種 FP8 方言；和 MI300X 的 vLLM（fnuz）對照 | 硬體 FP8 是 OCP 版 |
| LMDeploy TurboMind（int4／int8 KV、融合 attention） | 另一個業界 INT4 KV 實作 | CUDA |
| vLLM-KIVI（A100 調的 Triton kernel） | 在它原本設計的硬體上的速度 | 為 NVIDIA 調 |

- 我們自己的 Triton kernel（`m10_dequant_fast.py`、`m10_dequant_fast_x.py`、`m10_attn_q.py`）在 H200 上也能跑：**要重驗逐位元、重量時間**（11 問題一 §5 第 4 點：H200 的硬體 FP8 是 OCP e4m3，拆的成本可能和 MI300X 差很多，不能把 MI300X 的數字搬過去）。
- 這一節只是記下要用什麼；H200 要不要做、做哪些，還等使用者決定（10/17 還沒回覆第二階段要不要在 H200 復現）。

