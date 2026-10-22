# P0-1　重算速度的效度威脅：如果重算跟 vLLM 一樣快，結論方向還一樣嗎？

> ICPE 補量 P0-1（[README](README.md)）。平台 H200（hgx-cn01）。**§1–§6 是事先登記**，寫在任何 P0-1 的資料之前；結果寫在 §7 以後，登記那幾節不改（要改就在 §8「偏離」寫明時間與原因）。
> **登記寫於 2026-10-22T03:01:41Z（`date -u`），repo HEAD `cf31400`**；寫的時候沒有任何這個實驗的資料，程式也還沒寫。
> 標記：〔實測〕〔模擬〕〔算術〕〔判讀〕同 repo 慣例。

---

## 1. 問題

論文的三個主張都依賴 harness 的重算時間 f(i)：

1. 第一階段：寫入時就依位置決定放哪一層，沒有贏過「滿了才決定」的延後版（S5 對 S4+）。
2. 第二階段：每一層的最佳還原方式不同（CPU 層全部載入＋一次拆完、慢層用 Cake），G5b 選得準不準。
3. 3b：X1「猜誰會回來」的 oracle 空間、X5「精度×容量」的空間（全是模擬，用 f(i) 當重算線）。

可是 harness 比 vLLM 慢〔實測〕：

| 量測 | Llama-3.1-8B | Qwen2.5-7B-1M（noDCA） | 來源 |
|:--|--:|--:|:--|
| harness 只算 32K（s） | 2.12 | 1.865 | 14 號 M2 的 compute_only（RUNLOG_H200 job 21409 一節） |
| vLLM 預設切法 32K TTFT（s） | 1.2106 | 1.0590 | job 21409（guard 全 CLEAN） |
| 比值 | **1.75** | **1.76** | 同上；第一階段 C6（vLLM 冷啟動 1.224 s）是 1.74 |
| harness 的 f(i) 總和（64 塊，GPU 閒） | 2.0873 | NOT_MEASURED（這份檔沒有總和欄；f 在 `speed_qwen/bw.json`） | `results/m10_precision_h200/speed_llama/bw.json` |

harness 用 PyTorch SDPA（`m7_model.py`：`F.scaled_dot_product_attention` 加 lower-right 的 causal 遮罩，`enable_gqa=True`）。vLLM 在 H200 上用 FlashAttention 3（第一階段 R2-C6 的 vLLM log：`Using FlashAttention version 3`）。

**要證明的**：重算變快（用 H200 上最快的 attention 實際重量，另加一個「跟 vLLM 一樣快」的模擬版）之後，上面三個主張的方向不變；翻轉的格子逐格列出。

---

## 2. 步驟 S1：找出 H200 上最快、而且算得對的 attention（GPU，量時間）

### 2.1 候選（事先列死）

| 代號 | 實作 | 呼叫方式 |
|:--|:--|:--|
| `sdpa` | 現在的路徑（PyTorch SDPA＋`causal_lower_right`） | 不改，逐位元和原本相同 |
| `fa2` | vLLM 內附的 FlashAttention 2（`vllm.vllm_flash_attn.flash_attn_varlen_func(..., fa_version=2)`） | K、V 直接用 KV 緩衝的轉置視圖（不複製） |
| `fa3` | vLLM 內附的 FlashAttention 3（同上，`fa_version=3`；vLLM 在 H200 上用的就是它） | 同上 |
| `fa4` | vLLM 內附的 FlashAttention 4（CuTe DSL），只有 `is_fa_version_supported(4)` 在 sm_90 回 True 才試 | 同上 |
| `sdpa_cudnn` | PyTorch SDPA 的 cuDNN 後端（`sdpa_kernel(SDPBackend.CUDNN_ATTENTION)`），lower-right 的 causal 用實體化的布林遮罩 | 遮罩每個 (n, end) 快取一份 |

- **不放的**：FlashInfer——這個環境裡它的 JIT 編譯不過（第一階段 R2-C6，2026-10-16：nvcc 13.4 與 cccl 13.3 不相容，`#error "CUDA compiler and CUDA toolkit headers are incompatible"`）；照 FP8 凍結以外的 NV 移植原則也不改套件版本。記 NOT_MEASURED。
- 開關：環境變數 `M7_ATTN`（沒設＝`sdpa`＝原本的程式路徑）。只在 `m7_model.KVModel.forward_span` 的 attention 那一行換掉，其他（投影、RoPE、MLP、KV 緩衝、切法 512）都不動。`key_keep`（14 號丟 token）路徑不改，仍用 SDPA。
- 任何候選載入或呼叫失敗：完整錯誤記下、該候選出局，不修（規則 2；修會變成「另一個實作」）。

### 2.2 正確性閘（每個模型、每個候選）

- 輸入：固定 seed 的隨機 token（seed 0，範圍同 `m7_calib.c1`），Llama 8K、32K；Qwen 8K、32K。
- 參考：HF 整段 prefill（`KVModel.hf`，`attn_implementation="sdpa"`）最後一個位置的 logits。
- 候選：harness 以 512 一塊依序 prefill（`prefill_chunked`）後，最後一個位置的 logits。
- **過**＝argmax 相同、top-5 集合相同、max|Δlogit| ≤ max(0.25, 2 × `sdpa` 候選自己的 max|Δlogit|)。（第一階段 R1 對 HF 的最大差：Llama 8K 0.125、Qwen 8K 0.078、Qwen 32K 0.125。）
- 只報告：KV 緩衝和 `sdpa` 版的最大差（第 0 層的 K、V 與 attention 無關，應該逐位元相同；當作接線檢查，不同就停）。

### 2.3 速度與選擇

- 量法＝C1（`m7_calib.c1`）的協定：每個候選先暖機 4 塊，再依序量 chunk 0–63（32K）每塊的 f(i)（前後 `cuda.synchronize`），3 輪；**候選的順序每輪輪換**（輪 k 從第 k 個候選開始），減少時間漂移。
- 分數＝Σ_{i<64} 每塊 3 輪的中位數。
- **勝者**＝過了正確性閘的候選中分數最小的（每個模型各自選）。
- **停損**：勝者的分數沒有比 `sdpa` 小 ≥5% → 判「harness 已經用到最快可用的 attention」，S2(a)(b) 不跑（沒有東西可以重跑），S2(c) 只跑 V2；照實寫。
- 只報告：勝者 32K 的只算時間 ÷ vLLM 預設 TTFT（上表）＝新的「harness ÷ vLLM」。

### 2.4 用勝者重量的校準（GPU，量時間；都設 `M7_ATTN=<勝者>`）

| 項目 | 指令（和原本的 H200 run 相同，只加環境變數、改輸出目錄） | 原本的 H200 run |
|:--|:--|:--|
| C1（Llama，chunk 0–79，3 輪，GPU 閒） | `m7_calib.py c1 --max-chunks 80 --reps 3` | `20261016-061311-m7h-c1` |
| calib-f（Llama、Qwen） | `m10_speed.py calib-f --model <m>` | `20261017-074151-m10h-calibf-llama`、`-074209-…-qwen` |
| C5（GPU 忙的倍數；只量 H200 已選定的 duty 0.6、1.0） | `m7_write_policy.py busy-calib --duties 0.6 1.0` | `20261016-061331-m7h-c5` |
| C4（雜訊，t1：same ×1、restart ×10） | `m7_write_policy.py noise --reps 10 --slot t1 --proc-mode same／restart` | `20261016-061430-…`、`…-061447`～`…-061627` |

輸出：`results/icpe_h200/p01/calib/`。「GPU 忙」的定義不變：duty 1.0 的背景 matmul（H200 的 busy2）。勝者在 duty 1.0 下量到的變慢倍數只拿來當 V1 模擬的忙時倍數。

---

## 3. 步驟 S2：用新的 f(i) 重跑

### 3.1 (a) 第一階段延後測試，主設定（GPU，量時間）

- 指令：和 R3-B-main（`20261016-063929-m7h-b-nfs-gap0`）相同：`M7_IO_MODEL=share python code/m7_write_policy.py b --ssd-dev nfs --gap 0 --wl-seed 0 --verify`，只改 `--reps 6`（rep 0–5，一個 run 跑完；和合併組一樣是 6 次）＋環境 `M7_ATTN=<勝者>`、`M7_F_CSV=<2.4 的 C1>`（策略用的 f 和實際重算同一個實作）、`M7_RES=results/icpe_h200/p01/b`。cpu_frac 0.25／0.5／1.0、8 個策略、限速層＝MI300X 的 `tier_params.json`（照舊）。
- δ\* ＝ max(5%, 2.4 的 C4 δ)，算法同 R3（`m7_repro_compare.py`）。
- **判準（逐 cpu_frac）**：S5 對 S4+（回來請求 TTFT 中位數，rep 0–5 合併，`m7_analyze.b()` 同一套；正＝S5 較快）≤ +δ\* ⇒「寫入時決定仍沒有贏過延後版」。三個 cpu_frac 都成立才算 (a) 成立。
- 只報告：同一格的平均值版本（R-brr 發現 24 個請求的中位數在這個設定下會因兩群分布跳約 8%）、S5 對最好的簡單策略、和合併組（舊 f）的並排。

### 3.2 (b) 第二階段還原格子（GPU，量時間）

- 格子（24 格）：模型 {Llama, Qwen} × 層 {CPU 層（真實 H2D）、b32（Llama 2.49、Qwen 用它自己 bw.json 的 b32 GiB/s，**沿用舊值**，不因 f 變而重算）、NFS 級（0.331084 GiB/s＝`tier_params.json` 的 nfs 讀）} × GPU {閒, 忙（duty 1.0）} × 存法 {E0＝全 BF16, E1＝全 INT4（各模型 `fmt_choice.json` 的 int4 格式；拆法＝P3＝一次拆完，`m10_dqsweep.py env S4`）}。長度 32K。
- 每格量三種：**Cake**（`claim`）、**全部載入**（`loadall`）、**G5b**（`M10_MEET=g5b`，13 §8c 的選擇規則；它自己的線上 profiler 會用到當下的重算時間）。
- 指令：`m10_speed.py grid --model <m> --fmt <int4> --configs E0 E1 --bw-names cpu b32 nfs --gpu-states idle busy --extra-modes loadall claim --refs --lengths --launches 1 --reps 5 --launch0 <L>`（`--refs`、`--lengths` 不給值＝不跑參考線與其他長度）；bw.json 用一份複本，只加 `targets.nfs`。
- **配對設計**：同一個 job 裡 `sdpa` 與勝者兩個 arm 交錯：sdpa L0、勝者 L0、sdpa L1、勝者 L1（launch0＝500、501；兩個 arm 用同一組 seed）。每個 arm 每格 2 launch × 5 rep＝10 次。
- **每格的最佳方式**＝{Cake, 全部載入} 中還原時間中位數較小的；兩者差 ≤5%（相對較小者）記「平手」。
- **判準**：勝者 arm 和 sdpa arm 比，**翻轉**＝一邊是 Cake、另一邊是全部載入（兩邊都不是平手）。24 格 0 個翻轉 ⇒ (b) 成立。和平手之間的變化（「弱變化」）另列、不算翻轉。
- 只報告：每格 G5b ÷ 最佳、G5b 選全部載入的比例、32K 的 G5b 平均離最好（H200 舊值 11.5%）；sdpa arm 和舊的 43c（job 21109）同格的差（重現檢查）。

### 3.3 (c) 3b 的 X1、X5 模擬換常數（CPU）

三個版本，**只換重算線的常數**（f(i) 與 GPU 忙倍數）；限速層、位元組、反量化、H2D、CPU 層校正比值、trace、策略、判定程式都不動：

| 版本 | f(i) | GPU 忙倍數 |
|:--|:--|:--|
| V0 | 現在的 H200 常數（`calib_c1.csv`、`speed_<m>/bw.json`） | 現在的（1.945；X5 用它自己的 compute_only 比值） |
| V1 | S1 勝者重量的（2.4 的 C1、calib-f） | 2.4 的 C5 在 duty 1.0 量到的倍數（X1、X5 都用它） |
| V2 | 「跟 vLLM 一樣快」：V0 的 f × (vLLM 預設 32K ÷ harness 只算 32K)＝Llama × 0.5710、Qwen × 0.5678（上表） | 和 V0 相同 |

- X1：照 H200 原本的 runner（`r_h200_p3.sh` 的 X1-SWEEP、X1-ANALYZE）跑主 sweep＋判定；另跑截尾版（`-H1800`、`-H3600` 的 sweep 與 `x1_post.py` 的判定；H200 上還沒跑過，V0 也一起跑）。
- X5：照 runner 的 X5-GRID、X5-ANALYZE。
- 輸出：`results/icpe_h200/p01/x1_<V>/`、`x5_<V>/`（不覆寫 `results/p3_explore_h200/`）。V0 先和 git 裡的 `results/p3_explore_h200/{x1,x5}` 比（應逐值相同，ts／run_id 除外；不同就停下來查）。
- **判準**：V1、V2 各自的判定字和 V0 相同——X1 原登記（rule=min）、X1 截尾 H＝3,600 s、X5 登記判定。都相同 ⇒ (c) 成立。
- 只報告：X1 截尾 H＝1,800 s 的判定字；每格 headroom 的變化；跨過 15%／5% 門檻的格子逐格列出；V2 和第二階段報告 §5.5 估計（`whatif_recompute.json`）的對照。

---

## 4. 總判定

- **成立**＝(a)、(b)、(c) 都成立：重算跟 vLLM 一樣快時（V1 實測、V2 模擬），論文的主要結論方向不變。
- **不成立**＝任一條不成立；照實寫是哪一條、哪幾格翻了。不重新定義判準、不換統計量。
- S1 停損觸發時：寫「最快可用的 attention 仍是 SDPA」，(a)(b) NOT_MEASURED（不是不成立），(c) 只用 V2 判。

## 5. 停損與失敗處理

- 任何指令失敗：完整錯誤記進 RUNLOG，停下來回報（規則 2）。
- gpu_guard 標污染的 run：結果不進 `results/`，重量一次；第二次還污染就停下來回報。
- 量時間的 job 不和我們自己的任何 job 同時在 hgx-cn01 跑（開跑前檢查 `squeue -u $USER -w hgx-cn01`，有就退出）。CPU 模擬 (c) 也不和量時間的 job 同時跑（同一台主機）。

## 6. 時間預估〔估計〕

S1 約 20 分；2.4 約 30 分；(a) 約 2 小時（R3-B-main 3 rep 是 67 分）；(b) 約 1 小時；(c) 約 1 小時（16 CPU）。GPU 合計約 4 小時。

---

## 7. 結果

（還沒有。）

## 8. 偏離登記

（還沒有。）
