# 13　第二階段主線在 H200 上的復現（事先登記）

> 回到 [README](README.md)。MI300X 的凍結版：[07](07_freeze_mi300x.md)；MI300X 的結果：[08](08_report_mi300x.md)；原始規劃：[03](03_plan_h200.md)。
> **寫入時間**：2026-10-16T22:53Z（`date -u`），h200 分支 commit `944497e` 之後。**到這時 H200 上沒有任何第二階段的結果**（`results/m10_precision_h200/` 不存在；H200 只跑過第一階段、R1 的正確性與 CPU 自測）。MI300X 的第二階段結果已經公開在 08，所以這一頁是「看過 MI300X 的答案、沒看過 H200 的答案」時寫的。
> 這一頁寫死之後要改任何一項：另起一節寫「改了什麼、為什麼、在看到什麼結果之前或之後」，不覆蓋原文。

---

## 0. 要回答什麼

**MI300X 主線的每一個判定，換到 H200 上還一樣嗎？** 主線＝08 §0 的判定流程：
K0（FP8）→ Step 1（FP8 端點，兩個模型）→ Step 2b（換 codec）→ K3 → INT4 端點（K0、品質、速度不輸）→ INT4 Step 2（混合配置）→ K2。

H200 在論文裡的角色是「第二台硬體的佐證」。它不分擔 MI300X 的佇列，也不和 MI300X 的數字放進同一個判定。

## 1. 和 MI300X 完全相同的（不重選、不重調）

| 項目 | 值 | 怎麼確認相同 |
|:--|:--|:--|
| 判準 | 07 的全部：ε（配對差 CI 下界 > −2 個百分點；每族、每輪點估計 ≥ −5）、bootstrap（以題重抽、10,000 次、seed 0）、檢定力規則與 07 §9 的釐清、速度不輸（≤ 最快的混合配置 ×(1＋δ\*)）、K0／K1／K1s／K3／K2、穩定差距 | 同一支 `code/m10_analyze.py`（最後修改 commit `c34c250`，H200 沒改） |
| 程式 | 第二階段主線的 `code/m10_*.py`、`m7_restore_harness.py`、`m7_model.py`、`m7_busy.py`、`gpu_guard.py` | 和 `origin/main`（`bcfe5b1`）逐字相同，只差 `m10_tasks.py` 的模型路徑可以用 `M7_LLAMA_GLOB`／`M7_QWEN_GLOB` 指定（沒設時和 MI300X 一樣）。每個 job 開頭再比一次，不同就停 |
| 題目與難度 | `CANDIDATE_PARAMS.json`（07 §2 的 sha256）、正式題目 seed 10000 起、T2–T4 各 180、T5 60、主判準 T2–T5 | job 開頭驗 sha256 |
| codec | `m10_codec`：FP8 一律 e4m3fn（最大 448），兩台都不用 MI300X 原生的 e4m3fnuz；INT4、INT8 都是一般的 PyTorch 運算 | 程式相同；H200 的 GPU 自測（§3 V0）要過 |
| 配置與讀法 | E0、E1、E1s、E1sr、E1s@fp8k0_<粒度>、E1s@int8_token、F25／M25／B25／R25；讀法 A＋B（b32，B-閒／B-忙／B-閒→忙） | 指令照 `code/m10_queue_mi300x/queue/` 的 30、33b、33c，只換路徑 |

**§5.2 的 attention 修正：H200 復現也不做。** 03 開頭（10/15 晚）寫的是「H200 復現時再做」；這裡改成不做，理由是修了就改到 `m7_model.py`，H200 和 MI300X 跑的就不是同一份程式，判定不同時分不出是硬體還是程式造成的。修 attention 是另一個實驗。

## 2. H200 自己量的（照 MI300X 同樣的程序）

| 項目 | 程序 | 為什麼要重量 |
|:--|:--|:--|
| E0 難度確認 | seed 2000 起（MI300X 第 3 輪同一段 seed），兩個模型 × T1–T5 各 60 題，只跑 E0 | 03 §3.2、§7 D1：「H200 上只用 E0 確認仍在 60–95%」 |
| K／V 分布 | `m10_kv_stats.py`（只報告） | 03 開頭 10/15：「H200 上重量一次，報告只用 H200 的數字」 |
| 格式小測 | 07 §3 的程序：seed 4000 起、T2–T5 各 5 題，E0＋5 種 E1，只看平均 KL | 使用者 10/17：在 H200 照同一條規則重選 |
| 頻寬 | `m10_speed.py calib-f`：讓 E0 的會合點落在 block 16／32／48 | 重算速度 f(i) 是硬體決定的 |
| δ\* | `m10_speed.py noise`（E0，4 頻寬 × 閒／忙，3 process × 10 次） | 07 §4 的定義，每台各一個 |
| GPU 忙 | `results/m7_write_policy_h200/busy_levels.json` 的 busy2（duty 1.0，重算慢 1.945 倍；第一階段 R2-C5 在 H200 量的） | MI300X 是 duty 0.7（慢 2.07 倍）；同一個目標「約慢 2 倍」 |
| 會合點 | 速度格子（FP8、Step 2b、INT4），讀法 B 用 H200 的 `meets.json` | 讀法 B 的「哪些 block 來得及」取決於 H200 的速度 |

## 3. 停下來的條件（寫死；停了就回報，不換方法硬跑——規則 2）

1. **V0**：`m10_verify.sh`（完整模式）任何一項 FAIL。
2. **程式不同**：job 開頭比對 §1 的檔案和 `origin/main` 的 `bcfe5b1`，多出 `m10_tasks.py` 那段以外的差異。
3. **E0 難度**：任一個主判準任務族（T2、T3、T4、T5）在任一個模型上，E0 的合併答對率（兩輪合併；T1–T4 用 `score_64`，T5 用 `score_up`＝120 token）不在 60–95%。T1 只報告。
4. **格式**：H200 選到的 FP8 粒度或 INT4 分組，任一個和 MI300X 不同（MI300X：Llama fp8_token、Qwen fp8_head、INT4 兩個都 int4_kivi）→ 停，問使用者（10/17 決定）。
5. **速度格子**：`gpu_guard` 的 JSON 不是 CLEAN（自己這張卡上出現外來 process）→ 照 CLAUDE.md §3 整批重量；不改用沒標污染的那一半。
6. 任何指令 exit ≠ 0。

## 4. 先跑、封存

和 07 §6 同一條規則：判定流程還沒走到的配置可以先跑，結果放 `quality_sealed/`，走到才分析；走不到就寫「跑了、沒分析」。

- INT4 端點（E1／E1s／E1sr @INT4）：兩個模型都跑（MI300X 也是）。
- **INT4 混合配置（F25／M25／B25／R25 @INT4）：兩個模型都跑。** MI300X 的主線只在 Llama 上跑（33bz 判 Step 2 在 Llama），Qwen 是只報告的 46。H200 兩個都先跑，是因為兩個模型分在兩個 job、同時跑，Llama 的 job 不能等 Qwen 的結果才決定；判定照 H200 自己的 `decide_int4.json` 指定的模型，另一個只報告。

## 5. 和 MI300X 怎麼比

- **只比判定節點**：K0 每一格（贏／輸）、Step 1 每個模型（過／不過、哪一條）、K3、INT4 的 K0、INT4 端點每個模型（品質過／不過、速度不輸過／不過）、Step 2 的模型、K2。
- 每個節點寫「相同」或「不同」。**全部相同＝主線判定在 H200 重現**；有不同就逐一列出，不改判、不重跑到相同為止。
- 數字（答對率差、還原時間、會合點）並排列出，但不放進同一個判定或同一張圖的同一軸（03 §1：兩台的數字不混用）。原因：BF16 的運算 kernel 兩台不同，E0 本身就會有少數題答案不同（文獻：只換 GPU 型號，BF16 greedy 的答對率就會變，Yuan et al., NeurIPS 2025）。

## 6. 時間數字的限制

Slurm 的 job 裡看不到同節點別人的卡，`gpu_guard.host_contention` 一律回 `UNOBSERVABLE_HOST`。和第一階段在 H200 上一樣：
- 時間仍然用在 **H200 自己的**判定（K0、δ\*、速度不輸），每個 run 記鄰居快照（`tiara.sbatch` 的 `cotenants_*`），報告寫明「同節點的主機記憶體與 PCIe 爭用只能部分觀測」。
- **我們自己的 GPU job 不和量時間的 job 同時跑**（第一階段 F2 疊到 B 的教訓）：量時間的 job 單獨一張卡，品質 job 用 `--dependency=afterok` 接在後面。品質 job 在同一張卡上同時開多個 process，`gpu_guard` 會把自己的其他 process 記成 intruder → 該 run 的 `latency_ms` 作廢、分數保留（CLAUDE.md §3）。

## 7. 執行

| job | 內容 | 卡 | `--time` |
|:--|:--|:--|:--|
| `m10h-speed` | V0 驗證 → E0 難度 → K／V 分布 → 格式小測（停在 §3-4）→ calib-f → noise → FP8 格子＋K0 → Step 2b 格子 → INT4 格子＋K0 | 1 | 24 h |
| `m10h-q-llama`、`m10h-q-qwen` | Step 1（FP8 端點＋2b，讀法 A＋B）→ INT4 端點 → INT4 混合配置；每張卡同時開多個 process | 各 1，接在 `m10h-speed` 後面 | 40 h |

- 腳本：`code/r_h200_m10.sh`（`--part speed|quality`、`--from`、`--dry-run`、`--list`），工作池 `code/h200_pool.py`（看 `nvidia-smi` 的剩餘記憶體才開下一個），判定 `code/h200_m10_analyze.sh`（CPU，head node；包同一支 `m10_analyze.py`）。
- 結果：`results/m10_precision_h200/`；記錄：`results/RUNLOG_H200.md` 的 Milestone 10-H200；原始 run：`$TIARA_RUNS`。
- 不在這一批：41／43 大掃描、42–42c 反量化速度、44／44a 兩層與長 context 速度、70 vLLM TTFT（都在量時間，而且是 MI300X 的探索線）；44b／44c、45／46／55／60 的讀法 A 只報告、75 RULER CWE（只跑 E0）→ 主線結果出來後再決定。

---

## 8. 10/17 改動（使用者；寫的時候 H200 第二階段還沒有任何結果，job 20701–20703 還在排隊）

- **範圍擴大成「MI300X 量過的全部」**：使用者 10/17：「目前目標是把 mix 那邊量測的都跑過一遍，復現」，包括大掃描。§7 最後一段「不在這一批」的工作全部改成要做，清單與分批見 RUNLOG_H200 Milestone 10-H200 的「全部復現的清單」。
- **接受程式和 MI300X 不同**：使用者 10/17：「可以接受不同程式碼」。§3-2 的程式比對只留給已經送出的主線 job（20701–20703；它們的程式本來就和 MI300X 相同）。之後的 job 只**記錄**和 `bcfe5b1` 的差異（slurm 輸出），不擋。每個改動仍然是「加參數、預設不變」，並寫進 RUNLOG。
  - 第一個改動：`m10_sweep.py parts` 的 GPU 忙原本寫死 duty 0.7（MI300X 的 busy2），加 `--busy-duty`（預設 0.7）；H200 用自己的 busy2＝1.0。
- **128K 先跑**：使用者 10/17：「改先跑 128」。長 context（10_longctx_plan）排在主線品質之前：
  - `m10h-lc-speed`（量時間，1 張卡，接在 m10h-speed 後面）：大掃描的零件（`m10_sweep parts`，讀法 B 推算要用）→ 128K 的 FP8／INT4 速度格子 → 64K 的。
  - `m10h-lc-screen`（量分數，1 張卡，接在 lc-speed 後面）：試跑（三個長度）→ 照 10 §3.3 的規則決定題數（H200 的試跑時間）→ 正式，252 → 128 → 64 個 block。
  - 主線品質 job（20702、20703）改成接在 lc-speed 後面（量時間的 job 不和自己的 job 同時跑），而且等 lc-screen 開始之後才開始（讓 128K 先拿到卡）。
  - 程式：`code/r_h200_lc.sh`、`code/h200_longctx_plan.py`（MI300X 的 `longctx_plan_n.py`，規則一字不改，只換路徑與 run 名稱；試跑缺讀法 B 的輸入時停下，MI300X 版只印）。
- **256K（使用者問 32／64／128／256）**：還沒排。H200 上在 CPU 查過（寫這節之前）：
  - Llama-3.1-8B 的 `max_position_embeddings` 是 131,072：256K 要開 RoPE 外插，CLAUDE.md §8 禁止 → Llama 不做 256K。
  - Qwen2.5-7B-Instruct-1M 的 `dual_chunk_attention_config.original_max_position_embeddings` 是 262,144：這個長度以內是一般的 attention（我們的 HF 路徑不做 DCA），可以做。用 508 個 block（260,096 token，和 128K 一樣留 2,048 給問題與答案）。
  - 題目：508 個 block 的 T2、T3、T4 產得出來、`_check_item` 通過；**T5（cwe）產不出來**（`RuntimeError: cwe 詞表不夠長，補不滿`）。
  - 要做的話是 H200 才有的新實驗（MI300X 沒做），要另外登記；等 128K 的結果出來再問使用者。

## 9. 10/17 改動二（使用者「都照你建議的」；寫的時候 H200 第二階段只有 20701 在跑、還沒有任何品質或速度結果出來）

- **41 海選的題數固定 n＝24**（MI300X 的 `sweep_plan_n.py` 選的值、全部 12 種放法 × 17 種佔比），不在 H200 重算：兩台的題目才完全相同。09 §4.1 的「≤ 12 小時」在 H200 只當預算參考；兩個模型分在兩張卡。
- **43 複賽的選手用 H200 自己海選的結果**，照 09 §4.2 同一條規則選（`m10_sweep_analyze.py`）；最後並列兩台的選手與重疊。
- **75 的上游 RULER jsonl 在 H200 重新產生**：RULER `c3f5e3b4…`（和 MI300X 同一個 commit）、LFS 詞表 sha256 `affcd6d4…2eca`（和指標檔記的相同）、shim 只有 wonderwords＋tenacity 9.2.1；`$TIARA_DATASETS/ruler_ref/README.txt`。產生器的 seed 42 和 MI300X 相同；兩台的題目是否相同，用逐題的 gold 和 MI300X 的 `ruler_cwe_upstream_llama.csv` 比對（只比題目，不比分數）。
- 腳本：`code/r_h200_x.sh`（`--part q3 --model <m>`、`t2`、`q5`；開跑時才讀 repo 裡的版本，所以 t2、q5 送出後補的程式在開跑時生效）。

## 10. 10/18 K3 分岔之後怎麼走（使用者選「走 H200 的路＋對照」；寫的時候 Qwen 的 FP8 混合配置 `fp8mix_qwen_*` 還在 20707 裡跑，沒有任何一列被分析過）

**發生了什麼**（RUNLOG_H200「主線判定」）：K0、Step 1、2b 的 ε 和 MI300X 相同；Qwen 的 K3 不同——E1s@int8_token 在 CPU 層、GPU 忙那一格，H200 比最快的混合配置慢 12.3%（δ* 5%）→ 速度不輸不過；MI300X 慢 7.3%（δ* 9.4%）→ 過。照 §5「不改判、不重跑到相同為止」。

**H200 的路（判定）**：照 03 §2.5 第 4 步——FP8 階 Qwen「均勻不夠」→ 不往 INT4，在 FP8 階的 Qwen 上做 Step 2（§1.1）。
- 資料：stage 45 的 FP8 混合配置（`quality_sealed/fp8mix_qwen_*`：F25／M25／B25／R25 @fp8_head、讀法 A／B、b32、`meets_fp8tier.json`、T2–T4 n＝180、T5 n＝60；和 Step 1 同一批題、同 seed）。§4「先跑、封存」：判定走到了才分析，現在走到了。
- 分析：MI300X 寫好、凍結的 `code/m10_queue_mi300x/analyze_step2.sh` 換 H200 路徑（品質對 E0；穩定差距第 1、2 條；`--tag seedA`）。速度用 `analysis/fp8/speed_qwen.json`（δ* 0.05）與 `speed_qwen/summary_fp8.csv`（H200 沒有重量的那份）。
- 之後：K2 成立 → 位置策略停。穩定差距第 1、2 條成立 → 第 3 條要換新 seed 重跑一批（另一個 GPU job，到時再問）。
- 20707 的 fp8mix 任何一個 process 沒成功或 gpu_guard 報自己的卡上有外來 process：分數照用（品質規則，CLAUDE.md §3），列出來。

**對照（只報告，不是 H200 的判定）**：用封存的 INT4 資料照 MI300X 的路算一遍——INT4 的 decide、K0（INT4）、端點每個模型、route int4 指定的模型做 Step 2 → K2。輸出放 `analysis/compare_mi300x_path/`，和 H200 判定的 `analysis/fp8/` 分開；RUNLOG 和報告一律標「對照：假設 K3 成立」。

## 11. 10/18 加做三件（使用者「可以」；寫的時候這三件都還沒有任何資料）

### 11.1 Qwen 的 vLLM TTFT 用拿掉 DCA 的變體補量（70，只報告）
- 原因：20912 的 `TTFT-qwen-default` 失敗：vLLM 0.28.0 在 NVIDIA 上沒有可用的 DCA 路徑（`FlashAttentionImpl.__init__() got an unexpected keyword argument 'layer_idx'`），和 CLAUDE.md §3 在 3090 上記的一樣。
- 變體：`~/models/Qwen2.5-7B-Instruct-1M-noDCA`。權重、tokenizer 等全部 symlink 到原 snapshot `e28526f7…` 的 blob，只有 `config.json` 拿掉 `dual_chunk_attention_config`（chunk_size 262144、local_size 8192、original_max_position_embeddings 262144），其他 key 逐字相同。做法和 3090 時期的 `make_nodca_model.py`（commit `52803bc` 刪掉前）相同。
- 為什麼等價：TTFT 量 8K／16K／32K，都小於 DCA 的 chunk_size 262144，DCA 在一個 chunk 內就是一般的 attention；harness 的 transformers 5.19 Qwen2 沒有 DCA（`dual_chunk` 出現 0 次），所以 m10 的品質、速度本來就是這個 attention。
- 跑法：同 MI300X 的 70（default、512 兩種切法，`m10_vllm_ttft.py` 不改），只把 `M7_QWEN_GLOB` 指到變體；`vllm_meta.json` 記實際的 model_path。報告一律註明「Qwen 用拿掉 DCA 的變體」。MI300X 的 70 用什麼，等它跑了再並列。
- 量時間：和我們自己的其他 job 錯開（排在 t2 之後、q5 之前）。

### 11.2 256K（只有 Qwen，T2–T4；10_longctx_plan 的延伸，探索、不做判定）
- 長度：508 個 block × 512＝260,096 token（加生成留的空間 ≤ 262,144＝Qwen 拿掉 DCA 後的訓練長度）。Llama 上限 131,072，不做（CLAUDE.md §8 不准外推）。T5（CWE）在這個長度出不了題（10/17 試過）→ 只做 T2–T4。
- 速度：同 44a（`m10_speed grid --main-length 260096`，FP8 主線粒度 fp8_head、INT4 int4_kivi，主線 8 配置，主線的 bw.json，GPU 忙用 H200 的 busy_levels，2 launch × 5 rep），gpu_guard 要 CLEAN。
- 品質：同 44b／44c 的程序（試跑每任務 1 題 → 題數規則 → 正式），只有 Qwen × 508 個 block × T2–T4；放法、佔比、讀法、生成同 128K。題數規則同 10 §3.3（n ∈ {12, 9, 6, 3}，估計 ≤ 14 h；公式的任務數用 3）。工作池的 need 先設 70 GB（Qwen 252 個 block 在 H200 量到 peak 35.5 GiB；試跑記實際 peak）。
- 輸出：`results/m10_longctx_h200/speed_qwen_260096_<fmt>/`、`screen_plan_256.json`；分析併入 `m10_longctx_analyze`（256K 一欄，標「只有 Qwen、T2–T4」，和 32K–128K 的四任務平均不能直接比）。

### 11.3 δ* 用 5% 的對照（CPU，只報告，不改判）
- 原因：Llama 的 δ*＝0.284 撐住了 Step 1 Llama 的 K1，以及對照路徑的 INT4 Llama E1sr（RUNLOG_H200「H200 的判定走完＋對照」）。
- 做法：把 `speed_llama/noise.json` 的 δ* 換成 0.05（＝03 §5.3 的下限，也是 MI300X Llama 的值），其他輸入全部不動，用同一支 `h200_m10_analyze.sh`（`all`，加上需要時的 `compare`）在副本上重走。輸出 `analysis/sensitivity_delta005/`，README 標「只報告；H200 的判定照 δ* 0.284」。

## 12. 10/19 復現 MI300X 的 13 號（還原第二輪）與 14 號（載入／計算配對、丟 block）（使用者：「繼續復現所有實驗，能提交都提交，讓 H200 最大化」；寫的時候 H200 上這兩號都沒有任何結果）

**MI300X 的狀況（origin/main `c27ec61`，10/19）**：13 號的 43a–43e、44_v 有結果（RUNLOG_MI300X Milestone 14）；44cz／44d、14 號（44e0–44ez）排在 44ca／44cb 後面，還沒跑。

**範圍**：`43a_rs2_smoke 43b_rs2_busy 43c_rs2_g5 43d_rs2_newfmt 43e0_rs2_attn_resmoke 43e_rs2_attn 44cz_rs2_plan44d 44d_rs_longctx`（13 號）＋`44e0_md_smoke 44e1_ds1_screen 44e1z_ds_pick 44e2_ds2_confirm 44e3_d3_speed 44e4_m2_fmt 44e5_m1_unit 44ez_md_report`（14 號）。

**怎麼跑**：每個 stage 用 MI300X 的佇列腳本，經 `code/h200_port_stage.py` 轉成 H200 版，由 `code/r_h200_rs2.sh` 依序執行。轉換只換以下幾類，指令、題數、設定、判讀程式都不動：
- 路徑；
- run 名稱（m10h-）；
- vLLM 用的 python；
- 43d 讀的 43 腳本，換成 H200 自己的選手（同 §9）。

轉完檢查沒有殘留的 MI300X 路徑。失敗處理照 MI300X 腳本自己的設計：寫 NEEDS_HUMAN、跳過那一段、exit 0（RUNLOG_MI300X 14.5）。腳本本身 exit ≠ 0 才停（規則 2）。

**平台造成、事先寫死的差別**：
1. **P4a／P4b（ROCm `HSA_CU_MASK` 留運算單元）在 NVIDIA 上沒有這個開關。** 43a 的 `cumask` 照樣量，預期判成「沒生效」，43b 就照 13 §3.1 拿掉 P4，H200 寫 NOT_MEASURED（平台）。冒煙檢查原本要求背景「有傳遮罩」，改成只有遮罩生效（`OK_cumask`）時才要求（`m10_restore2.py smoke-check`；MI300X 有 OK_cumask，行為不變）。CUDA MPS 的 thread percentage 是不同機制，這一輪不做。
2. **背景強度掃描（13 §3.3）**：busy1、busy2「同主線」＝H200 主線的 duty 0.6／1.0（`results/m7_write_policy_h200/busy_levels.json`）。MI300X 的 busy3（duty 0.9，「更忙的方向」）在 H200 不跑：H200 的 busy2 已經是 duty 1.0。
3. **其他所有 GPU 忙的格子**用 H200 的 busy_levels（busy2＝duty 1.0，慢 1.945 倍）：`m10_speed.py`、`m10_attn_q.py` 的 `--busy-levels` 預設改成可讀 `M10_BUSY_LEVELS`（沒設時不變）。
4. **K\* 與 42c 冠軍用 H200 自己的**：`results/m10_sweep_h200/dqsweep/A/choice_<m>.json`；`pick-c` 讀 H200 的 42c 報告，預期是 S2k16（RUNLOG_H200）。Q4 的 P2 照 13 §3.1 的定義仍是 S2k8（方法本身，不換成 H200 的冠軍）。
5. **43e (d) vLLM 的 FP8 KV decode**：Hopper 的 FP8 是 OCP e4m3、硬體轉換（MI300X 是 fnuz），兩台的 vLLM FP8 不是同一種實作，只並列不比判定。Qwen 一開始就用拿掉 DCA 的變體（§11.1）；MI300X 是失敗後用 44_v 重跑，所以 H200 不另跑 44_v。
6. **δ\***：各報告用 H200 主線凍結的 noise.json（Llama 0.284、Qwen 0.05），照 13 §2「用主線凍結的」。
7. **43ca**（MI300X 一格被標污染後的重量）不跑；H200 自己的格子如果 gpu_guard 不是 CLEAN，照 CLAUDE.md §3 另外重量，到時記。
8. **44cb**（MI300X 同一個容器裡兩張卡的干擾檢查）不在這次範圍：H200 的 Slurm job 本來就各自一張卡、和別的 lab 共用節點。要不要做 H200 版（決定量時間時能不能同時跑我們自己的品質 job），另外問使用者。

**排程（量時間不和我們自己的其他 GPU job 同時跑）**：
- `rs2`（量時間，13 號）→ `md`（14 號的冒煙＋DS1／DS2，只量分數）→ `mdt`（量時間，14 號的 D3／M2／M1＋報告）；用 afterany 串起來。
- 三個 part 共用工作目錄 `$TIARA_RUNS/_m10h_work/rs2`，裡面放旗子、hold/、NEEDS_HUMAN。

**輸出**：
- `results/m10_restore2_h200/`、`results/m10_drop_h200/`、`results/m10_match_h200/`；
- DS 的逐列 CSV 放 `$TIARA_HOME/drop_m10h`（大，不進 git）。

**和 MI300X 怎麼比**：13、14 號都只報告，沒有判定節點。並列以下三類，數字不混用（§5）：
- 各段的挑選結果（`q4_pick`、`g5_pick`、`c_pick`、DS2 的選手）；
- 描述性結論的方向（例如「GPU 忙時 S4＋全載入幾乎不受背景影響」）；
- 各格的中位數。

**CPU 測試（head node，10/19）**：MI300X 的 `tests/restore2_cpu_shim.py`、`tests/match_drop_cpu.py` 在 H200 的環境 ALL PASS；全部 stage 轉換後 `bash -n` 過、沒有殘留路徑；`plan-44d` 用假的挑選結果寫出 40 個 grid，轉換後語法正確、K\* 是 H200 的。
