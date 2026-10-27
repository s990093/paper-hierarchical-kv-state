# E2E-C：組合策略的端到端評估（INT4＋Cake＋打字提示預取＋MPS 預留）

> **性質**：判準登記（事先寫死）。登記時間見文末「登記」一節（`date -u`）。登記之後這一節以上不改；和登記不同的做法寫成文末的「偏離 N」。
> **來源**：10/28 的草稿（同一個檔，原名 `E2E_combo_DRAFT.md`，commit 393b762）＋使用者 10/28 的決定：「好都做掉」（草稿 §6 的四點照建議）＋「模型都跑」（第 4 點的「只做 Llama」改成 Llama、Qwen 都做）。
> **為什麼有這一關**：ICPE 的故事缺一個正面的方法。手上的正面結果都是分開量的（INT4：ICPE D2；打字提示預取：C2〔模擬〕；MPS 預留：B0）。**SM 控制的端到端 TTFT、以及它在整條 trace 上對背景的代價，都還沒量**。
> **範圍**：只有 H200（使用者 10/28）。**第一段只做模擬**（CPU、head node）；第二段 GPU 驗證只在第一段判 E2E-GO 或 E2E-ASK 時做，而且要在任何 GPU 數字之前另外登記（§9）。

---

## 1. 現有的東西（讀程式確認過，10/28）

- ICPE X1 的 session 引擎（`code/p3x/x1_sim.py`）：CPU 層存 BF16、慢層存 INT4、還原用 Cake、預取的塊以 INT4 放在 CPU；HINT-PF（`code/icpe/x1_gpu.py::HintEngine`）、O-PF 已有。GPU 忙只有一個合成倍數（2.07），**沒有 MPS、沒有背景吞吐**。
- B0（`results/s3_h200/b0/raw/b0_{llama,qwen}_b0_*.csv`，32K context、64 塊）：背景 × 共用方式下的 `recompute1`、`dq_int4`、`h2d_perblock`、`e2e_cake_{bf16,int4}_{cpu,b32}` 的時間與背景吞吐比。
- B0 的 MPS 等級限制的是**背景 process**；`bg_ratio` 的分母是**同一個上限下**背景單獨跑的速度（`b0.py::start_bgproc`）→ 不含上限本身的代價（§3 另算）。
- trace：Mooncake conversation（`/work/ChihChungHsu/hungweilai/datasets/traces/conversation_trace.jsonl`，git blob `5a371794…`＝Mooncake `d750cdd` 的 `FAST25-release/traces/conversation_trace.jsonl`，12,031 筆，和 ICPE 同一份）；FreeInference（`/work/ChihChungHsu/hungweilai/datasets/freeinference_agentic_trace`，rev `16d3c04d`，16 週，81 檔 sha256 全對，run `20261028-023841-s3-fi-dl`）。

## 2. 配置

| 代號 | CPU 層 | 慢層 | 預取 | 共用方式 | 意思 |
|:--|:--|:--|:--|:--|:--|
| K0 | BF16 | **BF16** | LRU（不預取） | stream | 純 Cake |
| K1 | BF16 | INT4 | LRU | stream | ＋把位元組變小（＝ICPE X1 的 LRU） |
| K2 | BF16 | INT4 | HINT-PF | stream | ＋打字提示預取（＝ICPE C2 的 HINT-PF） |
| **K3** | BF16 | INT4 | HINT-PF | **固定 MPS 等級 m\*** | **組合策略** |
| O | BF16 | INT4 | O-PF | m\* | oracle 預取（上界參考） |

- 還原一律 Cake（X1 的 `View.restore`，mode＝cake）。K0：`View(..., slow_bytes=BF16 位元組, dq=0)`，引擎的 `int4_bytes` 也設成 BF16 位元組（慢層寫入照 BF16）。
- 每個配置都在同一個背景下跑（§3）；K0–K2 的共用方式＝stream，K3、O＝m\*。

## 3. 背景與共用方式怎麼進模型

### 3.1 時間的倍數（基本動作）

對（模型 M, 背景 g, 共用方式 m）：
- T(a｜g, m)＝B0 該格所有列的 `t_med_s` 的中位數；參考＝T(a｜idle, stream)。
- r_f＝T(`recompute1`)／參考 → 重算 f(i) 乘上它（還原的重算與新一輪的 prefill 都是）。
- r_d＝T(`dq_int4`)／參考 → 反量化 d 乘上它。
- r_h＝T(`h2d_perblock`)／參考 → CPU 層的讀取頻寬除以它。慢層（限速層）的讀取頻寬不變；寫入不變。
- **可用**：該格與參考格的 `sigma_rel` 都滿足 2σ ≤ 5%（05 §2.1）；任何一個要用的倍數不可用 → 那個（M, g, m）NOT_MEASURED（K3 不能選它；如果是 stream，那個（M, g）的格子全部 NOT_MEASURED）。

### 3.2 分解檢查 D（決定用 3.1 還是 3.3；只用 B0 的資料、不看任何 E2E 結果）

- 對每個（M, g ∈ {cprefill, mixed}, m ∈ {stream, mps, mps87, mps75, mps50}）與動作 a ∈ {`e2e_cake_bf16_cpu`, `e2e_cake_int4_b32`}：用 X1 的 `View.restore`（L＝64 塊、全部在 CPU BF16，或全部在 b32 的 INT4；b32 頻寬用 B0 的 Llama 2.9404、Qwen 1.4414 GiB/s、k＝1）算 pred＝T_model(g, m)／T_model(idle)，對 meas＝T(a｜g, m)／T(a｜idle, stream)。
- **D 過**（每個模型分開判）：|pred／meas − 1| ≤ 0.10 的組合佔可用組合的 ≥ 80% → 該模型用 3.1。
- **D 不過** → 該模型改用 3.3。

### 3.3 備用：直接用端到端的比值

還原那一段（`end − t0`）乘上 r_e2e，新一輪的 prefill 乘上 r_f。r_e2e 依那次還原讀了什麼選：讀了慢層 → `e2e_cake_int4_b32`（K0 為 `e2e_cake_bf16_b32`）；沒讀慢層但讀了預取的 INT4 → `e2e_cake_int4_cpu`；其他 → `e2e_cake_bf16_cpu`。SSD 與 NFS 級都用 b32 的比值〔代用〕。

### 3.4 背景吞吐的帳

- 單獨跑的參考速度：stream → solo(g, stream)；MPS 各等級 → solo(g, mps)。solo＝該區塊所有列的 `bg_solo_rate` 中位數。
- 上限本身的代價 cap(g, m)＝solo(g, m)／solo(g, mps)（m ∈ {mps87, mps75, mps50}）；stream 與 mps 的 cap＝1。
- 共跑時的比值 ρ(g, m, a)＝該格 `bg_ratio` 的中位數。每個請求 r（所有種類）的 GPU 區間 I_r＝[t_r, t_r＋TTFT_r]；ρ_r＝和 3.3 同一套對應的 e2e 動作的 ρ；沒有還原（L＝0）的請求用 `recompute1` 的 ρ。
- 一個時間窗 W：φ＝|∪_r (I_r ∩ W)|／|W|；ρ̄＝Σ_r |I_r ∩ W|·ρ_r ／ Σ_r |I_r ∩ W|；**背景吞吐 B＝cap × (1 − φ·(1 − ρ̄))**（相對於「沒有上限、單獨跑」）。
- FreeInference：每週一個重播，B＝各週 B 以 |W| 加權平均。

## 4. 開發集／測試集與 m\*

- 每個重播（conversation 一個；FreeInference 每週一個）裡，**開發集＝到達時間在前一半的請求**（t < (t_min＋t_max)／2），**測試集＝後一半**；時間窗 W 也照這樣切。指標只算回來的請求（`kind＝cont`）；FreeInference 照 X1 的規則把各週加總（Σ 總和 ÷ Σ 個數）。
- **m\*（M, g）**：候選 {mps, mps87, mps75, mps50} 中可用的；只留「開發集 B ≥ 0.95 在該（M, g）的 8 個開發格（2 trace × 2 慢層 × 2 CPU）都成立」的；取開發改善 (K2 − K3(m))／K2 在 8 格的中位數最大的；平手取上限較寬的（mps ＞ mps87 ＞ mps75 ＞ mps50）。一個都沒有 → 該（M, g）的 K3 ≡ K2（改善 0）。

## 5. 判準（測試集）

- **格子**＝（模型 {Llama-3.1-8B, Qwen2.5-7B-1M} × trace {conversation, FreeInference} × 慢層 {SSD, NFS 級} × CPU {128, 256 GiB} × 背景 {cprefill:2048:65536, mixed:512:32:8192}）＝**32 格**。
- 主指標：回來請求的平均 TTFT（不含排隊，同 X1 與 ICPE P0-2）。
- 一格**過**：Δ＝(K2 − K3)／K2 **> 15%**，而且 K3 的測試集 **B ≥ 0.95**。
- **E2E-GO**：過的格子 **≥ 8**（＝1/4），而且兩個模型**各至少 1 格**。
- **E2E-ASK**：不是 GO，但「Δ ≥ 5% 且 B ≥ 0.95」的格子 ≥ 8 → 停下來問。
- **E2E-NOGO**：其他 →「MPS 預留在端到端的好處不成立」，寫成量測結論。
- NOT_MEASURED 的格子算不過、照實列出；NOT_MEASURED 超過 8 格 → 判 **NOT_DETERMINED**（不判 NOGO）。
- **只報告（不判定）**：K0→K1→K2→K3 每一步的增量；K3 對 K0；O 的上界與 K3 拿到的比例 (K0 − K3)／(K0 − O)；K0–K2 的 B；B ≥ 0.90 的版本；P95；FreeInference 的使用者發起子集；開發集的數字。

## 6. 打字提示的提前量（HINT-PF 用）

- conversation：`x1_gpu.leads`（ICPE C2 的定義）；Qwen 用 `cpt_qwen_English` 與 Qwen 的 f(i)。
- FreeInference：照 X3（ICPE C1）的定義。使用者發起：min(打字時間, gap)，打字時間＝max(typed − 10, 0)·cpt ÷ (51.56·5／60)，typed＝本次與上一次的 `role_tokens["user"]` 之差；工具觸發：gap（工具呼叫開始時就知道它會回來；X3 的 `c_tool`）；gap＝start_r − end_parent（秒），< 0 → 0。parent＝`x1_fi` 的 parent。載入時斷言：和 `x1_fi.prepare_fi` 的請求順序、到達時間逐筆相同（CLAUDE.md 規則 6）。

## 7. Qwen 的 X1 常數（新加；Llama 用 ICPE 原本的 `x1_sim.constants()`，不改）

- f(i)：`results/m10_precision_h200/speed_qwen/bw.json` 的 `f_ms_median_by_chunk`（64 個），i ≥ 64 用全部 64 點的最小平方直線外插（同 `m8_trace_oracle.load_f` 的寫法）。
- 每塊位元組：BF16 29,360,128、INT4 9,175,040（`iowall.json`）；載入時斷言 BF16＝28 層 × 4 KV head × 128 × 2 × 2 B × 512（從模型 config 算，規則 6）。
- d：H200 的 Qwen INT4 格子（`20261017-103540-m10h-grid-int4-qwen`；E1、load_only、cpu、fp8_read > 0、閒）的 t_dequant／fp8_read 中位數（同 Llama 的規則）。
- 限速層、CPU 層：和 Llama 相同（MI300X 的 tier_params，ICPE 的規則）。Qwen 每塊比較小，同樣的 CPU 容量放得下比較多 session，這是模型本身的差異。

## 8. 事先講明的假設與限制

1. INT4 在降級時的編碼成本不算（同 ICPE X1）；W0 量到 CPU 壓縮跟不上，所以這個假設要求編碼在 GPU 上做，成本 NOT_MEASURED。
2. MPS 的上限在 process 啟動時固定（靜態預留）；背景在整條 trace 上都以 B0 的強度一直跑。
3. B0 在 32K context 量；X1 的 context 更長 → 假設倍數與長度無關〔未驗證〕。
4. X1 不排隊（同 ICPE）；縮小版 GPU 重播的排隊不代表真的 server（B6 的既有限制）。
5. SSD、NFS 級的端到端比值用 b32 代用（只在 3.3 用到時）。

## 9. 第二段（GPU 驗證）

只在第一段判 E2E-GO 或 E2E-ASK 時做；做之前另寫「附錄登記」：縮小 k、背景 process（B0 的 `start_bgproc`，MPS）、配置、次數、實測÷模擬的門檻（同 B6：[0.9, 1.1]、排序相同）。在附錄登記之前不跑任何 GPU 的 E2E。

## 10. 執行

- 程式：`code/s3/e2e.py`（新檔；import `x1_sim`、`x1_gpu`、`x1_fi`，不改它們）：`dcheck`（§3.2）、`run`（全部格子）、`report`（§4–§5）、`--self-test`（背景＝閒時新的 View 和原本的 `View` 逐請求逐位元相同；HINT-PF 的提前量在 conversation 上和 `x1_gpu.leads` 相同；FreeInference 的對齊斷言）。
- 輸出：`results/s3_h200/e2e/`。環境：`P3_CONST=h200`、`M7_RES`＝H200 的 m7、`X1_TRACE_DIR`、`X1_FI_DIR` 指到上面兩份資料。

## 11. 已經看過的數字（登記前）

- B0 的全部結果（RUNLOG S3-B0、`b0_report.json`），以及 10/28 為了可行性數過的「可用格子數」（Llama 144／175、Qwen 150／175）。
- ICPE 的 X1、C2、P0-2 的結果（ICPE_EVIDENCE B1–B7、C1–C2、D2）。
- O9（RUNLOG_H200 job 21402）。
- **沒有看過任何 E2E 的數字**（沒有跑過 §3.2 的檢查、沒有跑過任何配置）。

---

## 登記

- 登記時間（`date -u`）：2026-10-27T18:43:56Z
- 登記時 repo HEAD：`393b762`（本檔隨下一個 commit 進 git）

## 偏離（登記之後）

### 偏離 1（2026-10-27T18:50:37Z；看過 §3.2 的 D 檢查、沒有任何 E2E 配置的數字）：D 沒有任何可用組合 → 當作「D 不過」，兩個模型都用 3.3

- **看到的**：run `20261028-024941-s3-e2e-dcheck`（`results/s3_h200/e2e/dcheck.json`）：兩個模型的可用組合都是 **0**。原因是 3.1 要用的 `dq_int4` 倍數，閒時參考格（idle × stream）的 σ：Llama 0.124、Qwen 0.086（2σ 25%／17% > 5%）——就是 G-M11 的「閒時 INT4 融合 kernel 雙峰」（RUNLOG S3-B0、G-M11 結案）。
- **改什麼**：登記 §3.2 只寫了「可用組合 ≥ 80% 在 10% 內 → 3.1，否則 3.3」，沒寫可用組合為 0 的情況。照最保守的讀法：分解沒辦法被確認 → 不信基本動作的相乘 → **兩個模型都用 3.3**（端到端比值；3.3 不需要 `dq_int4` 的倍數）。程式本來就這樣處理（`frac`＝0 → `e2e`），這裡把它寫明。
- **3.3 下哪些不可用**（照 §3.1 的 2σ 規則，在跑任何配置之前查的）：Llama cprefill × mps50（`e2e_cake_int4_cpu` 2σ 39%）；Qwen cprefill × stream（`e2e_cake_int4_b32` 2σ 7.6%）→ 照 §3.1，Qwen × cprefill 的 8 格全部 NOT_MEASURED；Qwen cprefill × mps（同一個動作 2σ 11%）。NOT_MEASURED 共 8 格，沒有超過 §5 的 8 格 → 判定照常。
- **影響**：判準不變。另外記下一個程式的錯（不是偏離）：第一次 `dcheck`（run `20261028-024903-s3-e2e-dcheck`）因為 Qwen 時先換了 `x1_sim.BF`、再呼叫 ICPE 的 `constants()`（它斷言 Llama 的 BF）而失敗，沒有產生數字；已修。
