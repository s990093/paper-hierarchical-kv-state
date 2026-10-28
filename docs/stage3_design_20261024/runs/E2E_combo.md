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

---

## 結果（2026-10-28，第一段〔模擬〕）：**E2E-NOGO**

- **報告**：run `20261028-161142-s3-e2e-report` → `results/s3_h200/e2e/e2e_report.json`（指令：`source code/env_h200.sh; export P3_CONST=h200 TIARA_ALLOW_NO_SLURM=1; m7run s3-e2e-report python code/s3/e2e.py report`）。第一次報告 run `20261028-160552-s3-e2e-report` 的判定與 32 格逐值相同；第二次只多了 `m_star_candidates`（下面「m\* 的候選」那張表；**判定之後加的、只報告**，判定的程式沒有動）。
- **資料**：`20261028-025101-s3-e2e-run-llama-conv`、`20261028-025151-s3-e2e-run-qwen-conv`、`20261028-025305-s3-e2e-run-llama-fi`（14:22 結束）、`20261028-025307-s3-e2e-run-qwen-fi`（11:53 結束）；四個 exit 0、stderr 空；`results/s3_h200/e2e/raw/e2e_{llama,qwen}_{conversation,freeinference}.csv`＋`.meta.json`。D 檢查：`20261028-024941-s3-e2e-dcheck`（偏離 1）。
- **判定**：過 **0** 格（Llama 0、Qwen 0）；「Δ ≥ 5% 且 B ≥ 0.95」0 格；NOT_MEASURED 8 格（≤ 8，照常判定）→ **E2E-NOGO**：「MPS 預留在端到端的好處不成立」。照 §9，第二段（GPU 驗證）不做。
- **為什麼**：四個（模型, 背景）的 m\* 都選不出來（§4：要 8 個開發格的 B 都 ≥ 0.95）。**任何一個 MPS 等級、任何一個開發格的 B 都沒有到 0.95**（最高 0.944，Llama × cprefill × mps）；Qwen × cprefill 沒有可用的等級（偏離 1 的 NOT_MEASURED）。於是 K3 ≡ K2、Δ＝0。MPS 本身在開發集上讓 TTFT 改善很多（各等級 8 格的中位數 33–79%；8 格含 FreeInference 的 4 格，受下面的異常影響），擋下它的是背景吞吐：B 含上限本身的代價 cap(g, m) 和整條 trace 上前景占 GPU 的比例 φ（§3.4）；B0 的 SHARE（背景 0.98）是單一動作、不含 cap 的微基準。
- **不受 FreeInference 異常（下面）影響**：只看 conversation 的開發格，B 也都不到 0.95（同一張候選表的範圍）。

### 32 格（測試集；TTFT＝回來請求的平均，秒；B＝背景吞吐，相對「沒有上限、單獨跑」）

| 模型 | trace | 慢層 | CPU | 背景 | 判定 | Δ (K2−K3)/K2 | K3 的 B | K0 | K1 | K2 | O | K0 的 B | K2 的 B |
|:--|:--|:--|--:|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| llama | conv | ssd | 128 | cprefill | FAIL | 0.0% | 0.789 | 3.83 | 3.25 | 2.23 | 2.35 | 0.801 | 0.789 |
| llama | conv | ssd | 128 | mixed | FAIL | 0.0% | 0.687 | 1.50 | 1.05 | 0.65 | 0.65 | 0.682 | 0.687 |
| llama | conv | ssd | 256 | cprefill | FAIL | 0.0% | 0.790 | 3.48 | 2.99 | 2.29 | 2.37 | 0.798 | 0.790 |
| llama | conv | ssd | 256 | mixed | FAIL | 0.0% | 0.688 | 1.34 | 0.96 | 0.66 | 0.66 | 0.683 | 0.688 |
| llama | conv | nfs | 128 | cprefill | FAIL | 0.0% | 0.819 | 12.55 | 9.45 | 6.17 | 7.27 | 0.837 | 0.819 |
| llama | conv | nfs | 128 | mixed | FAIL | 0.0% | 0.711 | 5.04 | 3.03 | 1.96 | 2.32 | 0.707 | 0.711 |
| llama | conv | nfs | 256 | cprefill | FAIL | 0.0% | 0.814 | 10.28 | 8.17 | 5.27 | 6.12 | 0.831 | 0.814 |
| llama | conv | nfs | 256 | mixed | FAIL | 0.0% | 0.707 | 4.10 | 2.62 | 1.66 | 1.93 | 0.703 | 0.707 |
| llama | FI | ssd | 128 | cprefill | FAIL | 0.0% | 0.935 | 72.15 | 36.56 | 87.03 | 61.60 | 0.936 | 0.935 |
| llama | FI | ssd | 128 | mixed | FAIL | 0.0% | 0.926 | 26.31 | 10.50 | 26.73 | 18.49 | 0.924 | 0.926 |
| llama | FI | ssd | 256 | cprefill | FAIL | 0.0% | 0.938 | 27.55 | 22.28 | 66.31 | 28.64 | 0.940 | 0.938 |
| llama | FI | ssd | 256 | mixed | FAIL | 0.0% | 0.936 | 8.25 | 5.99 | 20.20 | 8.30 | 0.943 | 0.936 |
| llama | FI | nfs | 128 | cprefill | FAIL | 0.0% | 0.931 | 216.49 | 216.42 | 527.93 | 705.10 | 0.932 | 0.931 |
| llama | FI | nfs | 128 | mixed | FAIL | 0.0% | 0.907 | 85.52 | 67.94 | 169.51 | 224.18 | 0.904 | 0.907 |
| llama | FI | nfs | 256 | cprefill | FAIL | 0.0% | 0.936 | 55.78 | 51.82 | 214.36 | 338.23 | 0.938 | 0.936 |
| llama | FI | nfs | 256 | mixed | FAIL | 0.0% | 0.929 | 19.77 | 15.36 | 70.26 | 105.32 | 0.929 | 0.929 |
| qwen | conv | ssd | 128 | cprefill | NOT_MEASURED | | | | | | | | |
| qwen | conv | ssd | 128 | mixed | FAIL | 0.0% | 0.696 | 0.82 | 0.57 | 0.45 | 0.45 | 0.686 | 0.696 |
| qwen | conv | ssd | 256 | cprefill | NOT_MEASURED | | | | | | | | |
| qwen | conv | ssd | 256 | mixed | FAIL | 0.0% | 0.703 | 0.64 | 0.51 | 0.49 | 0.49 | 0.694 | 0.703 |
| qwen | conv | nfs | 128 | cprefill | NOT_MEASURED | | | | | | | | |
| qwen | conv | nfs | 128 | mixed | FAIL | 0.0% | 0.701 | 2.13 | 1.46 | 0.64 | 0.46 | 0.694 | 0.701 |
| qwen | conv | nfs | 256 | cprefill | NOT_MEASURED | | | | | | | | |
| qwen | conv | nfs | 256 | mixed | FAIL | 0.0% | 0.706 | 1.26 | 0.97 | 0.56 | 0.49 | 0.695 | 0.706 |
| qwen | FI | ssd | 128 | cprefill | NOT_MEASURED | | | | | | | | |
| qwen | FI | ssd | 128 | mixed | FAIL | 0.0% | 0.950 | 4.00 | 3.67 | 5.11 | 3.63 | 0.957 | 0.950 |
| qwen | FI | ssd | 256 | cprefill | NOT_MEASURED | | | | | | | | |
| qwen | FI | ssd | 256 | mixed | FAIL | 0.0% | 0.962 | 3.66 | 3.65 | 3.65 | 3.65 | 0.962 | 0.962 |
| qwen | FI | nfs | 128 | cprefill | NOT_MEASURED | | | | | | | | |
| qwen | FI | nfs | 128 | mixed | FAIL | 0.0% | 0.939 | 8.89 | 6.23 | 41.96 | 41.89 | 0.944 | 0.939 |
| qwen | FI | nfs | 256 | cprefill | NOT_MEASURED | | | | | | | | |
| qwen | FI | nfs | 256 | mixed | FAIL | 0.0% | 0.961 | 3.77 | 3.70 | 4.11 | 3.73 | 0.960 | 0.961 |

（m\* 全部選不出來 → 每格 K3＝K2；O 用 m\* 或沒有 m\* 時的 stream。）

### m\* 的候選（開發集；只報告，判定之後加進 `e2e_report.json` 的 `m_star_candidates`）

| （模型, 背景） | 等級 | 可算的開發格 | 開發集 B 範圍 | B ≥ 0.95 的格數 | 開發改善中位數 (K2−K3)/K2 |
|:--|:--|--:|:--|--:|--:|
| llama, cprefill | mps | 8/8 | 0.766–0.944 | 0 | 56.7% |
| llama, cprefill | mps87 | 8/8 | 0.667–0.779 | 0 | 74.5% |
| llama, cprefill | mps75 | 8/8 | 0.638–0.765 | 0 | 78.9% |
| llama, cprefill | mps50 | 0/8 | — | 0 | — |
| llama, mixed | mps | 8/8 | 0.613–0.908 | 0 | 37.7% |
| llama, mixed | mps87 | 8/8 | 0.595–0.849 | 0 | 38.0% |
| llama, mixed | mps75 | 8/8 | 0.584–0.832 | 0 | 40.1% |
| llama, mixed | mps50 | 8/8 | 0.449–0.638 | 0 | 45.7% |
| qwen, cprefill | mps | 0/8 | — | 0 | — |
| qwen, cprefill | mps87 | 0/8 | — | 0 | — |
| qwen, cprefill | mps75 | 0/8 | — | 0 | — |
| qwen, cprefill | mps50 | 0/8 | — | 0 | — |
| qwen, mixed | mps | 8/8 | 0.623–0.921 | 0 | 33.3% |
| qwen, mixed | mps87 | 8/8 | 0.599–0.874 | 0 | 33.8% |
| qwen, mixed | mps75 | 8/8 | 0.586–0.811 | 0 | 35.0% |
| qwen, mixed | mps50 | 8/8 | 0.467–0.635 | 0 | 44.3% |

### 只報告的項目（§5，不判定）

- **K0→K1→K2 的增量（conversation，24 格裡可算的 12 格）**：K0→K1（INT4）Llama 14.2–39.8%、Qwen 20.5–31.3%；K1→K2（打字提示預取）Llama 23.5–38.5%、Qwen 5.0–56.3%；K0→K2 Llama 34.3–61.0%、Qwen 24.4–70.0%。K3＝K2，所以 K3 對 K0 就是 K0→K2。
- **O 的上界比例 (K0 − K3)／(K0 − O)（conversation）**：0.89–1.21；12 格裡 10 格 ≥ 1（O-PF 騰空間照 LRU，不是上界；同 ICPE C2 的〔推論〕）。
- **B（conversation，stream）**：K0 0.68–0.84、K2 0.69–0.82。不加任何上限、只算前景占用，背景在 conversation 上就掉 16–32%。
- **B ≥ 0.90 的版本**：0 格過（m\* 是照 0.95 選的；K3＝K2 時 Δ＝0）。
- **P95（conversation，單一重播）**：K0／K2／O，例：Llama SSD 128 cprefill 11.76／6.10／6.07 秒、Llama NFS 128 mixed 14.22／6.98／9.10 秒、Qwen NFS 128 mixed 7.85／1.95／1.16 秒（逐格在 `e2e_report.json` 的 `p95`）。
- **開發集**：每格的開發集 TTFT／B 在 `e2e_report.json` 的 `dev`；開發集的 Δ 全部 0（K3＝K2）。
- **FreeInference 的使用者發起子集**：在 `ttft_user`（K2＝K3）。

### FreeInference 的異常（照實記；沒有驗證原因）

- **前 6 週和 ICPE 一致**：第 1–6 週（ICPE X1-FI 跑的就是這 6 週；回來請求數逐週相同，例：第 1 週 10,613）K0、K1、K2、O 的 TTFT 幾乎相同（例：Llama SSD 128 cprefill 第 1 週測試集 12.77 秒，四個配置都一樣）＝ICPE B3「FreeInference 沒有空間」。
- **第 7–16 週 ICPE 沒有跑過（X1-FI 的 sweep 每個設定只有前 6 週）；從第 9 週起負載重很多**（測試集每週回來請求 3.7–10.1 萬筆，第 1 週 0.53 萬），平均 TTFT 升到幾十到上千秒（例：Llama NFS 128 cprefill 第 16 週 K1 511.88、K2 1,149.21、O 1,354.46 秒），而且**預取（K2 的打字提示、O 的 oracle）比不預取更慢**。所以 32 格表裡 FreeInference 可算的 12 格，K1→K2 有 11 格是負的（−11% 到 −573%）、1 格 +0.2%；O 的比例沒有意義。
- **推論（沒有驗證）**：慢層通道被塞滿、請求在通道上排隊，預取再加慢層讀取的流量。X1 不模擬請求排隊，但慢層通道是逐筆排的。**這些數字不拿來下任何結論**；判定不受影響（見上）。
- 要不要查（例如加通道使用率的輸出、或只取前 8 週），由使用者決定。

### NOT_MEASURED（8 格，照偏離 1）

- Qwen × cprefill:2048:65536 的 8 格（2 trace × 2 慢層 × 2 CPU）：stream 的 `e2e_cake_int4_b32` 2σ 7.6%（σ 0.038169，參考 0.002244）；mps 同一個動作 2σ 11%。
- 另外 Llama × cprefill × mps50 不可用（`e2e_cake_int4_cpu` 2σ 39%），只少一個候選等級，格子照算。

### 意思

- 組合策略裡「固定 MPS 預留」這一條腿，在端到端、背景吞吐預算 0.95 之下不成立：它是用背景吞吐換前景 TTFT 的取捨，不是免費的。ICPE 的「不用預測、只要固定預留」的說法要改。
- 不用預測的另一條腿（INT4，K0→K1）和打字提示預取（K1→K2）在 conversation 上照舊有效〔模擬〕。

## 判定之後的補充（2026-10-28；只報告，判定照舊 E2E-NOGO）

### A. MPS 的取捨曲線（`code/s3/e2e_curve.py`，run `20261028-165753-s3-e2e-curve` → `results/s3_h200/e2e/mps_tradeoff.{csv,png}`）

測試集，每組 4 格（2 慢層 × 2 CPU）的中位數；改善＝(K2 − K3(m))／K2，B 同 §3.4。

- **基準線（K2、stream）本身就不到 0.95**：conversation 的 B 0.70–0.80（前景在整條 trace 上占 GPU 的比例 φ 很高），FreeInference 0.93–0.96。登記的「B ≥ 0.95」是絕對門檻，conversation 的開發格在任何共用方式下都到不了。這是 NOGO 的主要原因，登記時沒有先算基準線的 B。
- **不設上限的 mps 相對 stream**：TTFT 改善 33–63%，B 只變 0 到 −7 個百分點（Llama cprefill conversation 0.802 → 0.778；Llama cprefill FreeInference 0.935 → 0.936；Llama mixed conversation 0.698 → 0.631）。有上限的 mps87／75／50 多拿的改善不多，B 掉得多。
- **MPS 本身的開銷**：B0 的背景單獨跑的速度，cprefill 和 mixed 在 mps 與 stream 下相同（兩個模型 `solo(mps)/solo(stream)` 都是 1.000；cprefill 的速率只有每秒約 6 步，解析度約 8–17%）。所以 §3.4 用「MPS 下單獨跑」當分母，在這兩種背景下沒有低估代價。
- **限制**：FreeInference 的格子落在下面 B 節的異常週；全部是〔模擬〕（B0 的微基準倍數）；第二段 GPU 驗證沒有做。

### B. FreeInference 第 9–16 週異常的診斷（`code/s3/e2e_diag.py`，run `20261028-164518-s3-e2e-diag-fi` → `results/s3_h200/e2e/diag/fi_anomaly.csv`）

- 看之前先寫下的假設（程式開頭）：H1 慢層通道飽和；H2 CPU 層被預取擠得打轉；H3 §3.3 把等通道的時間也乘上背景倍數（只會放大）；H4 程式的錯。
- 做法：Llama、CPU 128 GiB、SSD／NFS，第 1 週與第 9 週（第 16 週見下），LRU／HINT-PF／O-PF × {cprefill 背景、沒有背景}，記 E2E 的 TTFT 與 X1 的 `summarize`。
- **第 1 週**：三個做法 TTFT 完全相同（有背景 12.77 s、沒有背景 1.52 s），慢層使用率約 0 → 和 ICPE X1-FI 一致。
- **第 9 週（測試集 TTFT，秒；有背景／沒有背景）**：

| 慢層 | LRU | HINT-PF | O-PF |
|:--|--:|--:|--:|
| SSD | 16.35／1.36 | 32.45／1.96 | 15.82／1.26 |
| NFS | 54.57／4.80 | 319.81／34.33 | 84.36／7.49 |

- **H3 排除**：沒有背景時 HINT-PF 一樣比 LRU 慢（NFS 4.80 → 34.33 s）。
- **H2 是主因**（NFS、沒有背景）：HINT-PF 讓回來請求從 CPU 載入的塊數從 16,748,257 掉到 7,468,589；從慢層載入的塊從 845,041 升到 3,131,555；重算的塊從 2,805,555 升到 9,798,709；降級次數從 4,545 升到 35,580。SSD 同方向（CPU 載入 16.77M → 8.68M、降級 4,545 → 46,412）。預取把很快要用的塊擠出了 CPU。
- **H1 是次要**：慢層讀的平均使用率 NFS 0.087 → 0.395（O-PF 0.205）、SSD 0.009 → 0.074；整週平均沒有到 1（尖峰沒有量）。
- **H4**：沒有證據支持。現象和 HINT-PF 的定義一致（每個有提示的 session 都預取，騰空間照 LRU，沒有准入控制）；O-PF 在 NFS 也變慢（4.80 → 7.49 s），同樣是騰空間的問題。沒有逐行驗證程式，不能完全排除。
- **意思**：第 9 週起同時在等工具結果的 session 多很多；HINT-PF 不管 CPU 壓力把它們全部預取，CPU 只放得下工作集的一小部分，於是把馬上要回來的塊擠掉。**免費的提示在高併發的 agent 負載下沒有准入控制會變成負的。** 這是〔模擬〕；ICPE 的 C2（conversation）不受影響，FreeInference 第 1–6 週也不受影響。
- **第 16 週沒跑完**：這個診斷在第 16 週占了 32 GB 記憶體，head node 只剩 3 GB 可用（Claude Code 也因此收掉了背景監看）→ 我停掉它（exit 143），第 16 週沒有寫進任何列。等 T1 的 FreeInference 跑完再單獨跑。
- **第 16 週（補跑；只跑這一週）**：`20261029-023519-s3-e2e-diag-fi-w16`（02:35–03:13，exit 0；同一條指令只留 `--weeks 2026-08-30_2026-09-05`，附加到同一個 CSV）。有背景的三個值和 E2E 報告相同（NFS 511.88／1,149.21／1,354.46 s），對得上。測試集 TTFT（秒；有背景／沒有背景）：

| 慢層 | LRU | HINT-PF | O-PF |
|:--|--:|--:|--:|
| SSD | 83.16／5.50 | 212.87／14.03 | 184.40／6.41 |
| NFS | 511.88／62.07 | 1,149.21／155.11 | 1,354.46／170.48 |

  - **H3 仍然排除**：沒有背景時預取一樣比 LRU 慢，連 O-PF 也是（NFS 62.07 → 170.48 s；SSD 5.50 → 6.41 s）。
  - **H2 仍在**（NFS、沒有背景）：
    - 回來請求從 CPU 載入的塊：41,998,352 → HINT-PF 10,394,033、O-PF 2,523,767。
    - 重算的塊：31,020,979 → 60,582,537／68,142,122。
    - 降級次數：59,668 → 148,890／181,048。
  - **H1 在第 16 週變成主要的一部分**：
    - LRU 自己的慢層讀平均使用率就有 0.612（有背景 0.650），預取後到 0.901／0.932（有背景 0.914／0.949）。
    - SSD 沒有背景時 0.187 → 0.436／0.515。
    - 慢層通道接近飽和，所以平均 TTFT 到幾十到上千秒。
  - **意思**：第 16 週的負載連 LRU 都讓慢層接近滿載。預取（就算是 oracle）沒有准入控制，同時佔 CPU 空間、又佔慢層通道，所以兩邊一起變差。結論和第 9 週相同、更強：這幾週的數字反映的是「模擬的慢層與 CPU 容量撐不住這個負載」，不是預取本身的價值。〔模擬〕
