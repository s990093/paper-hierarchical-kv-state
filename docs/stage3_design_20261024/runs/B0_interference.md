# B0＋G-M08：像真的服務的背景下，還原的各種做法被拖慢多少；分 SM／改 stream 優先權能換到多少

> **性質**：判準登記（事先寫死）。登記時間見文末（`date -u`）。之後這一節以上不改；不同的做法寫成文末的「偏離 N」。
> **來源**：[05 §5](../05_poc_plan_h200.md)（B0 的格子、GB0-FLIP／SHARE／STATIC）、[08 §3](../08_kill_argument.md)（prefill 節點背景、只用生產格式的子集、STATIC 的判讀）、[mechanisms/06 §1–§2](../mechanisms/06_eval_gates.md)（G-M08：長 context decode 背景）、[mechanisms/03](../mechanisms/03_bandwidth_compute.md) M08–M10。
> **平台**：只在 H200（hgx-cn01）。**量時間、獨佔**：節點上不能有我們自己的任何別的 job（含 DL_Lab4、ICPE、stage3 的品質 job；runner 開跑前檢查）；排在 ICPE 量時間的鏈之後（memory `stage3-gpu-scheduling`）。別人的 job 管不到：每個 run 記 `squeue -w`（tiara.sbatch 的 context／cotenants_end）。

---

## 1. 設定

### 1.1 回來的 session 與模型

| setup | 模型 | 回來的 session | 還原動作作用的範圍 |
|:--|:--|:--|:--|
| b0 | Llama-3.1-8B、Qwen2.5-7B-1M | 32K（64 塊，亂數 token，fa3 prefill） | 64 塊（端到端）／block 1–64（基本動作） |
| m08 | Llama-3.1-8B（128K＝252 塊）、Qwen2.5-7B-1M（256K＝504 塊） | 同上 | 一個單位＝16 塊（block 1–16） |

### 1.2 背景（`code/s3/bgengine.py`：和 vLLM 的 FlashAttention 後端同一條路——分頁 KV（page 16）、fa3 `flash_attn_varlen_func`、一步一個 CUDA graph；權重和回來的 session 共用）

| 名稱 | b0 | m08 | 類型 |
|:--|:--|:--|:--|
| idle | ✓ | ✓ | 對照 |
| decode:B:8192（B＝8、32；Llama 再加 64、Qwen 再加 128） | ✓ | | 頻寬型（真實） |
| mixed:512:32:8192（512-token prefill 段＋32 條 decode，同一個 varlen 呼叫） | ✓ | | 混合（真實） |
| cprefill:2048:65536（一直跑 2K 的 chunked prefill，接在 64K 之後；＝prefill 節點、＝新進來的長文件） | ✓ | ✓ | 算力型（真實） |
| decode:4:129024（Llama）；decode:4:258048、decode:6:258048（Qwen） | | ✓ | 長 context decode（真實） |
| matmul（4096³ BF16 ×8）、hbm（2 GiB D2D） | ✓ | | 舊的兩個極端（對照） |

「像真實服務的背景」＝decode、mixed、cprefill（05 §5.3 的判準只看這些）。背景單獨跑的速度每個區塊前量一次（2 秒）。

### 1.3 還原動作

| 動作 | 內容 | 生產格式子集 |
|:--|:--|:--|
| h2d_contig | 一次 copy 64 塊連續的 pinned 記憶體（copy engine） | ✓ |
| h2d_perblock | 逐塊 `copy_` 進 KV 緩衝（Restorer 的路徑） | ✓ |
| h2d_batch | `vllm._custom_ops.swap_blocks_batch`（cuMemcpyBatchAsync；每（塊, 層, K/V, KV head）一段 128 KiB） | ✓ |
| sm_gather_N | vLLM 的 Triton zero-copy kernel（SM 去讀 pinned 記憶體），N＝12（vLLM 的固定值）、32、132 個 program | ✓ |
| dq_int4 | INT4 KIVI 反量化（`m10_dequant_fast` 的融合 kernel；資料已在 GPU） | ✓ |
| dq_int4_torch | 同上，m10_codec 的 PyTorch 寫法 | ✓ |
| dq_fp8 | FP8（fp8_token）反量化，m10_codec 的 PyTorch 寫法（FP8 的 Triton kernel 照 memory `fp8-code-freeze-until-amd` 不動） | ✓ |
| dq_nvfp4 | NVFP4 反量化（SM90 沒有硬體 FP4，m10_codec_x 的軟體解碼） | ✓ |
| lr_recon | xKV 粗底 B16（`xl:k20:v30:cal+q:int4_kivi`，A0 登記 §1.3）的解碼：係數反量化＋乘基底＋RoPE | |
| recompute1 | 重算中間那一塊（fa3） | ✓ |
| qprefill | （m08）在 context 後面 prefill 512 個問題 token | ✓ |
| e2e_{cake,load}_{bf16,int4}_{cpu,b32} | （b0）端到端：`m10_speed.Bench` 的 Cake（claim）／全部載入，CPU 層（不限速、真的 H2D）／b32（H200 fa3 的會合點 b*＝32 反推的頻寬：Llama 2.9404、Qwen 1.4414 GiB/s，ICPE P0-1 S1 的 `bw.json`） | ✓ |

基本動作的正確性在每個 run 開頭核對（搬運＝原值逐位元；融合的 INT4 反量化＝PyTorch 版逐位元），不對就停。

### 1.4 共用方式

| 名稱 | 做法 |
|:--|:--|
| stream | 同 process：背景在自己的 stream（優先權 0），還原在另一條一般優先權的 stream（端到端：Restorer 的 stream 都是一般優先權） |
| stream_hi | 同上，還原的 stream（含 Restorer 的重算、搬運、拆）改成最高優先權 |
| mps | MPS daemon（job 自己的 pipe／log 目錄），背景在另一個 process、不限 |
| mps87／mps75／mps50 | MPS，背景限 87／75／50% 的 SM（`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`）；還原的 process 不限 |
| green | （只有 P0e 確認 green context 能用才做；做的話另登記）|

b0：stream、stream_hi、mps、mps87、mps75、mps50。m08：stream、stream_hi、mps、mps87、mps75。

### 1.5 量法

- 每個 (背景, 共用方式) 一個區塊：背景開始跑、穩定 0.5 秒後，動作照**隨機順序**（種子記在每列）各做 1 次暖機＋6 個時間窗。每個時間窗 ≥1 秒、內含 n_exec 次背靠背執行（每次之間同步、計時）；一列＝一個時間窗：執行時間的中位數、P95、最小值，以及時間窗內背景完成的步數÷秒（背景吞吐）。背景吞吐比＝時間窗內的速度÷區塊開始前單獨跑的速度。
- 背景的順序也隨機。每個 cell 的值＝6 個時間窗的中位數的中位數。
- δ\*＝5%（ICPE P0-1 S1 的 C4：δ 0.02–0.03% → max(5%, 雜訊)）。
- 污染：GpuWatcher 的結果照記；`contaminated` 的 run 不進 results、重量（CLAUDE.md §3）。

---

## 2. 判準（事先寫死）

- **GB0-FLIP**：存在一種像真實服務的背景，在它底下至少 2 條（模型 × 下層）線的最快端到端計畫（stream 共用方式下的 e2e 動作裡最快的）和「閒」時不同，而且「閒時最快的計畫」比這個背景下最快的慢 ≥10%。
- **GB0-SHARE**：存在（模型, 像真實服務的背景, 動作, 共用方式 ≠ stream），使還原比 stream 快 ≥30%，而且背景吞吐比 ≥0.90。
- **GB0-STATIC**：對每個（模型, 動作），在背景吞吐比 ≥0.90 的 MPS 等級（mps＞mps87＞mps75＞mps50）裡取還原最快的；如果每個（模型, 動作）在所有像真實服務的背景下的最好等級都在相鄰一級之內 → 固定預留就是答案（主張改成「該預留多少」的量測，不做線上規劃器；08 §1.2）。
- 讀法（05 §5.3）：FLIP、SHARE 都不成立 → B 不做線上選擇器，降成路 2 論文的一節量測；只有 SHARE → B1 只做「還原給多少 SM」。
- **G-M08**：m08 的長 context decode 背景下，存在（模型, 背景, 共用方式, 計算型動作 ∈ {recompute1, qprefill, lr_recon}），使動作的速度 ≥ 閒時（stream）的 60%（t_閒／t ≥ 0.60），而且背景吞吐比 ≥0.95（TPOT 慢 ≤5%）→ PASS；有資料但找不到 → STOP（「長 context decode 沒有可用的空檔」）。
- **另報（不判定）**：干擾矩陣（Paragon 式：每個動作對每個背景的拖慢倍數）；只用生產格式的子集時的排序變化（08 §3 第 2 點）；vLLM #42212 的靜態選路（28 KiB／12 SM）在同一格的表現（sm_gather_12 vs h2d_batch）；P95。

## 3. 成本〔估計，NOT_MEASURED〕

b0 每個模型：stream 兩種約 1 小時、MPS 四種約 2.2 小時；m08 兩個模型約 2.5 小時。合計約 9 小時，獨佔。

## 4. 和 05／06 不同的地方（登記前就決定；理由）

1. **Llama 的 decode 最大 B 用 64（不是 128）**：128 條 × 8K × 128 KiB＝128 GiB，加上權重放不進 141 GB 的 HBM（vLLM 在一張 H200 上也放不下）。Qwen（56 KiB／token）照 05 用 128。
2. **端到端的「漸進式（A）」不在 B0 量**：要先有 A0 的結果才知道用哪個配置；A2 再量。B0 的端到端只有 Cake 與全部載入（BF16／INT4）。
3. **共用方式沒有「時間片輪流」**（05 §2：只當舊數字的對照；舊數字已經有 13 號 Q4 的 P0）。
4. **green context**：torch 2.13 有 `torch.cuda.GreenContext`，能不能在這台用由 P0e（`b0.py env`）確認；能用的話另外登記一節再加，不在這次的判準裡。
5. **背景的 decode 位置固定**：每一步覆寫同一個 KV 位置（形狀固定才能錄成 CUDA graph）；每一步的計算量、讀的位元組和真的 decode 相同，但 KV 不會變長。

---

## 登記

- 登記時間（`date -u`）：**2026-10-25T12:37:58Z**；commit：`git log -- docs/stage3_design_20261024/runs/B0_interference.md` 的第一個 commit（訊息開頭 `docs(s3-b0): prereg`）。
- 登記時沒有任何 B0／G-M08 的數字：程式（`code/s3/b0.py`、`code/s3/bgengine.py`）只過了 CPU 自測，還沒在 GPU 上跑過。

---

## 偏離（登記之後）

### 偏離 1（2026-10-25T13:06:45Z；使用者要求；還沒有任何 B0／G-M08 的還原時間數字）：「雜訊」的操作定義 → δ\* 改成 05 §2.1
- **改了什麼**：§1.5 的「δ\*＝5%」改成 **05 §2.1**：每個量測點 δ\*＝max(5%, 2×σ_noise)，σ_noise＝那個比較的 baseline 一臂 6 個時間窗的相對標準差；判「有改進」要中位數差 > δ\* 且差的 95% CI（對 rep 重抽 10,000 次）下界 > 0；判「沒有變慢」要變慢 ≤ δ\* 且 95% CI 上界 ≤ δ\*；P95 用 P95 的 bootstrap 標準誤。2σ > 5% 的點先回報、不判定。
- **各判準怎麼套**（門檻本身不變，只加雜訊與 CI 的條件）：
  - GB0-FLIP：baseline＝那個背景下「閒時最快的計畫」；要 慢 ≥ max(10%, δ\*) 且 CI 下界 > 0。
  - GB0-SHARE：baseline＝同一格的 `stream`；還原要快 ≥ max(30%, δ\*) 且 CI 下界 > 0；背景吞吐「少 ≤10%」照舊（門檻 10%），另要背景吞吐比的 95% CI 下界 ≥ 0.90；背景單獨跑的速度改成量 6 個 1 秒的時間窗（σ 從這裡來）。
  - GB0-STATIC：「最好的等級」改成「和最快的差 ≤ δ\* 的等級都算並列最好」；存在一個等級，使每個像真實服務的背景都有一個並列最好的等級落在它的相鄰一級內 → STATIC。
  - G-M08：「TPOT 慢 ≤5%」改成「背景變慢 ≤ δ\*（σ 來自背景單獨跑的 6 個時間窗）且變慢的 95% CI 上界 ≤ δ\*」；「速度 ≥ 閒時 60%」的門檻不變（另報 CI）。
- **CSV 加欄位**：`n_rep`、`sigma_rel`、`delta_star`、`p95_sigma_rel`、`p95_delta_star`、`bg_solo_sigma_rel`、`bg_delta_star`；少於 6 個時間窗的點寫 NOT_MEASURED。
- **為什麼**：05 §2 原本只寫「δ\*＝max(5%, H200 上量到的雜訊)」，雜訊沒有定義怎麼量；這份登記直接取 5%。
- **受影響的 run**：全部 B0／G-M08 的 run（`b0-run`）。到這一刻都還沒跑：冒煙 22238 在量任何還原之前失敗（block 範圍越界）；冒煙 22242 正在跑（輸出在 run 目錄、不進 results、不判定）。看過的數字只有 P0b 背景自檢的單獨速度（每步毫秒），判準不用它。
- **補充（同日，還沒有任何時間數字）**：「沒有變慢」的點估計 ≤ δ\* 但 95% CI 上界 > δ\* → 那一格記 **INCONCLUSIVE**（要加重複次數），不算過也不算變慢。G-M08：有任何一格過 → PASS；沒有過、但有 INCONCLUSIVE → **INCONCLUSIVE**（先加重複次數再判）；都明確沒過 → STOP。GB0-SHARE 的「背景吞吐少 ≤10%」同樣：點估計過但 CI 下界 < 0.90 → 那一格 INCONCLUSIVE。

### 偏離 2（2026-10-25T13:33:09Z；使用者 10/25 選「G-M11 VQ（併進 B0）」；還沒有任何 B0 的時間數字）：還原動作加 `dq_vq`
- b0 與 m08 的動作清單加 `dq_vq`（查表式 VQ 解碼，定義與判準見 [G_M11_vq.md](G_M11_vq.md)）；它不在「生產格式子集」裡。開頭的正確性核對加一項「VQ 解碼＝碼本查表」。B0 自己的判準（FLIP／SHARE／STATIC／G-M08）不變。

### 偏離 3（2026-10-25T14:23:30Z；實作，判準不變；B0／G-M08 還沒有任何時間數字）：量測期間加 GpuWatcher，被污染的 run 不寫進 results

- **改什麼**：`code/s3/guard.py`（RUNLOG S3-GUARD-2）。量測開始前開監看（開不起來 → 這個 run 直接停，不在看不到的情況下量）；結束時被污染（有外來 process，或監看期間 nvidia-smi 查詢失敗過，S3-GUARD-1）→ 結果寫到 run 目錄的 `<檔名>.CONTAMINATED`、留 `CONTAMINATED` 檔，**不寫進 results**，要重量；每列多 `own_gpu_intruders`（0／外來者數／`UNOBSERVED:<n>`）；report 再按 run_id 排除一次（`excluded_contaminated_runs`）。B0 原本就有監看，但被污染的 CSV 留在 `results/s3_h200/b0/raw/`、report 也照讀，和本登記「contaminated 的 run 不進 results、重量」不一致；現在污染 → 這個 run 的 CSV 和 `_numa.json` 搬到 run 目錄。B0 冒煙 22242 的四個 gpu_guard.json 都是 CLEAN（MPS 模式在 runner 設了 `GPU_GUARD_JOB_CGROUP_OWN=1`，guard.py 用同一個 own_root）。
- **為什麼**：CLAUDE.md §3（gpu_guard 必用；contaminated 的 run 不得寫進 results）。原本的程式只靠 runner 開跑時的獨佔檢查，量測期間沒有監看（ICPE session 10/25 提醒）。
- **影響的 run**：全部（都還沒跑）。

### 偏離 4（2026-10-26T02:47:24Z；實作＋容量規則；還沒有任何 MPS 的時間數字——22275 在第一個 MPS 區塊就失敗，部分輸出沒看、已搬出 results）：MPS 下放不下的格子寫 NOT_MEASURED

- **為什麼**：MPS 模式的背景在另一個 process，要**自己再載一份權重**；stream 模式的背景和主程式共用權重。22275（llama b0 MPS）的第一個區塊 `decode:64:8192`：背景 process 80.12 GiB（權重約 15＋分頁 KV 64.12）＋主程式 59.27 GiB > 139.79 GiB → 主程式在 `lr_recon` OOM（之後才冒出 illegal memory access）。診斷 job 22389（MPS、只跑 lr_recon／recompute1、decode:8 與 decode:64）都正常 → 不是 lr_recon 的 bug，是容量。
- **改什麼**（`b0.py run`、`bgengine.step_bytes`）：有 MPS 模式時，量測前先把每個動作跑一次，取得主程式的峰值；每個 MPS 區塊開背景 process 前算「背景 process ≈ 權重＋背景常駐位元組（`step_bytes`，和 ModelStep 同一個算法）＋6 GiB 餘量」，加上「主程式還會長多少（峰值 − 現在）」，大於 `cudaMemGetInfo` 的可用量 → 這個（背景, 模式）的每個動作寫一列 `t_med_s=NOT_MEASURED`、`note` 寫原因，跳過。report 把這些格子列進 `not_measured`（不當成任何一邊的證據）。
- **影響**：只影響 MPS 的格子；判準不改。照算法，llama b0 的 `decode:64:8192` 在 MPS 下會是 NOT_MEASURED；其他格子放不放得下要看執行時的實際數字（照實記在 `note`）。stream／stream_hi 不受影響。CSV 多一欄 `note`（加在最後）。
- **補充（2026-10-26 05:0xZ，同一條偏離）**：可用量在 `cudaMemGetInfo` 之前先 `empty_cache()`（PyTorch 快取保留的記憶體不算已用）。22431（llama b0 MPS）用的是沒清的版本，`decode:64:8192` 那時記「可用 61.3 GiB」；清了也放不下（約 139.8 − 56.3 − context ≈ 82 < 85.5），結論相同。之後的 MPS run 用清過的版本。
