# 05　壓縮格式表（第三階段地圖實驗的輸入；只讀第二階段結果）

> 回到 [04 地圖實驗 v2](04_prereg_map.md)。這張表是 04 §1（格式）、§3（壓縮做法）、§5（成本常數、模型軸）的輸入。
> **狀態**：2026-10-18 做完第一版。只用 CPU、只讀既有結果；**沒有跑任何 GPU，也沒有重量 MI300X 第二階段的東西**。
> **規則**：每個數字都附 run_id 與來源檔（機器可讀的版本在 `results/p3_formats/format_table.json`，每一格都有 `{value, unit, status, run_id, src, method, note}`）；沒量到寫 NOT_MEASURED；第二階段還在跑、之後會有的寫 PENDING。

---

## 0. 一頁摘要

- **位元組**：30 列（2 個模型 × 15 種格式）全部用程式算出來，再和實際存下來的大小對拍：**每一列都逐位元組相同**（公式＝CPU 上真的 `encode()` 的 `Enc.nbytes`＝MI300X 存下來的大小＝3090 存下來的大小）。scale／zero 都算在內。
- **反量化**：MI300X 有四種量法（單獨量、kernel 微測、stage 40 bench、還原中），差很多（§3.2）。**給模擬器用的是「還原中」那一欄**，因為它含了「和重算搶 GPU」的成本。3090 只有單獨量（GPU 閒）和 Qwen 的還原中（主線寫法）。
- **品質（32K）**：生成答對率**有量到**的是 BF16、FP8 主線、FP8-K0、INT8、INT4-KIVI。除了 Qwen FP8（−32.8 分，FAIL），其他都過 2 分界線（四種情境都過）。**INT2-KIVI 和其他 8 種新格式只有 teacher forcing**（41 海選，96 題，沒有 CI）；生成品質要等 stage 43（只涵蓋部分格式）與 44c。
- **新算的數字**：Llama 的 FP8-K0、INT8 生成品質。第二階段沒報過，因為 Step 2b 只判 Qwen。這次用**凍結的** `m10_analyze.py` 在既有的列上補算，兩個重現檢查都逐欄相同（§4.1）。

## 1. 檔案與 run

| 東西 | 位置 |
|:--|:--|
| 表（寬表，每格三欄：值、`__run_id`、`__src`） | `results/p3_formats/format_table.csv`（30 列 × 269 欄，有 `run_id`、`ts`） |
| 表（巢狀，每格都有完整出處，另含對拍與 pending） | `results/p3_formats/format_table.json` |
| 待補清單 | `results/p3_formats/quality_pending.csv`（同時寫在 JSON 的 `quality_pending`） |
| 補算的 Llama FP8-K0／INT8 生成品質 | `results/p3_formats/derived/quality_llama_step2b{.json,_table.csv}`、`derive_quality.json`（重現檢查） |
| 程式 | `code/p3_formats/build_format_table.py`、`code/p3_formats/derive_quality.py` |

| run_id | 做什麼 | 結果 |
|:--|:--|:--|
| `20261018-002550-p3fmt-qderive` | 補算 Llama `E1s@fp8k0_token`、`E1s@int8_token` 的生成品質＋兩個重現檢查 | exit 0；重現檢查 2／2 逐欄相同 |
| `20261018-023324-p3fmt-table` | 建表（第一次） | **exit 1**：我自己的程式錯，見 §7 |
| `20261018-023339-p3fmt-table` | 建表（修好） | exit 0；之後又改了 PENDING 的標法，所以作廢、被下一個取代 |
| `20261018-023443-p3fmt-table` | **建表（這一版）** | exit 0；problems＝[] |

全部用記錄殼跑：`source code/env_3090.sh; TIARA_NO_GPU=1 bash code/run3090.sh p3fmt-<短名> python code/p3_formats/<…>.py`。run 目錄在 `/ssd8/hungwei/paper-hkv/runs/<run_id>/`。

## 2. 位元組

形狀（`m10_dqsweep.SHAPES`）：Llama 32 層 × 8 個 KV head、Qwen 28 層 × 4 個 KV head，D＝128，一個 block＝512 token。BF16：Llama 128 KiB/token、Qwen 56 KiB/token（和 04 §5 相同）。

**每個 block 的位元組有五個來源，30 列全部一致**（`bytes_measured_agree_with_code`＝True）：

1. `bytes_per_block_code`：照 `m10_codec.py`／`m10_codec_x.py` 每個 tensor 的形狀與 dtype 寫出的公式（`build_format_table.half_bytes`），含 FP32／FP16 scale、FP16 zero、E8M0／E4M3 的 block scale、NVFP4 每層的 FP32 全域 scale、FP8-K0 的 BF16 K 平均。
2. `bytes_per_block_encode_cpu`：這個 run 在 CPU 上用合成 BF16 block 真的呼叫 `m10_codec_x.encode(x, fmt).nbytes`。
3. `bytes_per_block_mi300x_parts`：MI300X 真實 KV 的 `Enc.nbytes`（`results/m10_sweep_mi300x/parts/parts_<m>.json`；run `20261016-130329-m10-sweep-parts-llama`、`-130451-…-qwen`）。
4. `bytes_per_block_mi300x_restores`：還原格子的 `bytes_stored`（E1 ÷ 64；E1s 先扣 1 個 BF16 block 再 ÷ 63），以及 41 海選 ALL 的 `bytes_stored`（同樣扣掉 block 0）。
5. `bytes_per_block_3090`：3090 的 `chunk_bytes`（BF16），Qwen FP8／INT4 用 X7 原始 restores 的 `bytes_read ÷ 64`。

**「另一種 FP8 粒度」的兩列**（Llama `fp8_head`、Qwen `fp8_token`）沒有存過的實體，只有公式＋encode，所以標 `code_only`。

per-token 的值＝每 block ÷ 512。per-head、per-layer 的 scale 是攤在 512-token 的 block 上算的。例如 Qwen `fp8_head` 是 8.0005 bit，不是 8.000。block 長度不同時，這部分會變。

## 3. 反量化（每個 block，ms）

### 3.1 主表（中位數；NM＝NOT_MEASURED、P＝PENDING）

各欄的定義：
- **MI300X 單獨**：parts（stage 40）用真實 KV，單獨量一個 block 反量化進暫存區，64 block × 3 次；GPU 忙＝背景 matmul duty 0.7；gpu_guard CLEAN。K0＝原寫法（PyTorch 一步一步），融合＝`m10_dequant_fast`（K1）。
- **MI300X 還原中**：真實 CPU 層、32K、Cake 的 E1（2b／43 用 E1s），每 block 的值＝median(`t_dequant` ÷ 這次讀到的低精度 block 數)。K0 的來源依格式：主線 FP8 用 `20261015-161226-m10-grid-fp8-llama`、`20261015-194150-m10-regrid-fp8-qwen`；INT4 用 `20261016-034902-m10-grid-int4-llama`、`20261016-052143-m10-grid-int4-qwen`；FP8-K0／INT8 用 2b 的 run；int2_kivi、mxfp4、K4V2、K8V4 用 stage 43 的速度格子（見 §3.3）。融合用 stage 42：`20261016-232855`／`-234030`／`-235027`／`20261017-000209`。
- **3090 單獨**：X7 `20261017-180859-p3x7-dqbench`（`m10_dequant_fast.py --self-test --bench`，GPU 閒，guard CLEAN）。
- **3090 還原中**：X7 `20261017-181306-p3x7-grid-qwen-L0`。從 run 目錄的原始 `restores.csv` 重算，1 launch × 5；host_level＝HEAVY，而且同 switch 的鄰卡有 PCIe 流量（X7 §4.4）。

**Llama**

| 格式 | B/block | KiB/token | 對 BF16 | 實測＝程式 | MI300X 單獨 K0 閒／忙 | MI300X 單獨 融合 閒／忙 | MI300X 還原中 Cake K0 閒／忙 | MI300X 還原中 Cake 融合 閒／忙 | 3090 單獨 K0／融合（閒） | 3090 還原中 Cake K0 閒／忙 |
|:--|--:|--:|--:|:--|:--|:--|:--|:--|:--|:--|
| bf16 | 67108864 | 128.000 | 1.0000 | 是 | 0／0 | 0／0 | 0／0 | 0／0 | 0／0 | 0／0 |
| fp8_token | 34603008 | 66.000 | 0.5156 | 是 | 0.2348／0.5949 | 0.1103／0.1743 | 0.3867／1.033 | 0.1037／0.3174 | 0.9759／0.172 | NM／NM |
| fp8e5_head | 33556480 | 64.004 | 0.5000 | 是 | 0.311／0.7722 | NM／NM | NM／NM | NM／NM | NM／NM | NM／NM |
| fp8k0_token | 34668544 | 66.125 | 0.5166 | 是 | 0.4267／0.7351 | NM／NM | 0.4096／1.1618 | NM／NM | NM／NM | NM／NM |
| int8_token | 34603008 | 66.000 | 0.5156 | 是 | 0.2338／0.5249 | 0.1256／0.2346 | 0.3819／1.0434 | NM／NM | 0.9748／0.1915 | NM／NM |
| kv-int8_token-int4_kivi | 27787264 | 53.000 | 0.4141 | 是 | 0.4397／0.6544 | NM／NM | NM／NM | NM／NM | NM／NM | NM／NM |
| int4_kivi | 20971520 | 40.000 | 0.3125 | 是 | 0.4962／0.6374 | 0.1484／0.1697 | 0.7122／1.8163 | 0.1559／0.4693 | 2.0531／0.2468 | NM／NM |
| int4_token | 17825792 | 34.000 | 0.2656 | 是 | 0.3669／1.247 | 0.1137／0.1271 | NM／NM | NM／NM | 1.6251／0.171 | NM／NM |
| int4h_token | 17825792 | 34.000 | 0.2656 | 是 | 1.8922／4.2032 | NM／NM | NM／NM | NM／NM | NM／NM | NM／NM |
| mxfp4 | 17825792 | 34.000 | 0.2656 | 是 | 0.5832／0.7935 | NM／NM | 1.0447／2.434 | P／P | NM／NM | NM／NM |
| nvfp4 | 18874624 | 36.001 | 0.2812 | 是 | 0.5822／0.7264 | NM／NM | NM／NM | NM／NM | NM／NM | NM／NM |
| kv-int4_kivi-int2_kivi | 16777216 | 32.000 | 0.2500 | 是 | 0.7019／1.1401 | NM／NM | 0.9796／2.0553 | P／P | NM／NM | NM／NM |
| int2_kivi | 12582912 | 24.000 | 0.1875 | 是 | 0.5726／1.408 | NM／NM | 1.4996／2.1866 | P／P | NM／NM | NM／NM |
| int2h_g32 | 12582912 | 24.000 | 0.1875 | 是 | 2.1498／4.5106 | NM／NM | NM／NM | NM／NM | NM／NM | NM／NM |
| fp8_head（另一種粒度） | 33556480 | 64.004 | 0.5000 | 只有程式 | NM／NM | NM／NM | NM／NM | NM／NM | 0.9923／0.1956 | NM／NM |

**Qwen**

| 格式 | B/block | KiB/token | 對 BF16 | 實測＝程式 | MI300X 單獨 K0 閒／忙 | MI300X 單獨 融合 閒／忙 | MI300X 還原中 Cake K0 閒／忙 | MI300X 還原中 Cake 融合 閒／忙 | 3090 單獨 K0／融合（閒） | 3090 還原中 Cake K0 閒／忙 |
|:--|--:|--:|--:|:--|:--|:--|:--|:--|:--|:--|
| bf16 | 29360128 | 56.000 | 1.0000 | 是 | 0／0 | 0／0 | 0／0 | 0／0 | 0／0 | 0／0 |
| fp8_head | 14680960 | 28.002 | 0.5000 | 是 | 0.1113／0.2917 | 0.0747／0.207 | 0.1725／0.4147 | 0.0533／0.1988 | 0.4557／0.1229 | 0.854／1.7865 |
| fp8e5_head | 14680960 | 28.002 | 0.5000 | 是 | 0.1516／0.3096 | NM／NM | NM／NM | NM／NM | NM／NM | NM／NM |
| fp8k0_head | 14709632 | 28.056 | 0.5010 | 是 | 0.1487／0.4002 | NM／NM | 0.2334／0.4865 | NM／NM | NM／NM | NM／NM |
| int8_token | 15138816 | 28.875 | 0.5156 | 是 | 0.1104／0.295 | 0.0735／0.2518 | 0.166／0.4085 | NM／NM | 0.4485／0.1178 | NM／NM |
| kv-int8_token-int4_kivi | 12156928 | 23.188 | 0.4141 | 是 | 0.2114／0.359 | NM／NM | 0.3187／0.7833 | P／P | NM／NM | NM／NM |
| int4_kivi | 9175040 | 17.500 | 0.3125 | 是 | 0.4859／0.3747 | 0.1361／0.1345 | 0.411／1.0375 | 0.0824／0.3505 | 0.9247／0.1485 | 1.9979／3.8925 |
| int4_token | 7798784 | 14.875 | 0.2656 | 是 | 0.1896／0.2344 | 0.0795／0.086 | NM／NM | NM／NM | 0.7055／0.087 | NM／NM |
| int4h_token | 7798784 | 14.875 | 0.2656 | 是 | 1.7772／2.0947 | NM／NM | NM／NM | NM／NM | NM／NM | NM／NM |
| mxfp4 | 7798784 | 14.875 | 0.2656 | 是 | 0.7569／0.8054 | NM／NM | NM／NM | NM／NM | NM／NM | NM／NM |
| nvfp4 | 8257760 | 15.750 | 0.2813 | 是 | 0.5462／0.6497 | NM／NM | NM／NM | NM／NM | NM／NM | NM／NM |
| kv-int4_kivi-int2_kivi | 7340032 | 14.000 | 0.2500 | 是 | 0.3358／0.8756 | NM／NM | 0.6526／1.1591 | P／P | NM／NM | NM／NM |
| int2_kivi | 5505024 | 10.500 | 0.1875 | 是 | 0.5924／0.463 | NM／NM | NM／NM | NM／NM | NM／NM | NM／NM |
| int2h_g32 | 5505024 | 10.500 | 0.1875 | 是 | 1.7292／1.4144 | NM／NM | NM／NM | NM／NM | NM／NM | NM／NM |
| fp8_token（另一種粒度） | 15138816 | 28.875 | 0.5156 | 只有程式 | NM／NM | NM／NM | NM／NM | NM／NM | 0.4495／0.1208 | NM／NM |

CSV／JSON 還有這幾欄，上表沒放：
- MI300X 只載入（load_only）的 K0／融合；
- 42b A 段 kernel 微測（`results/m10_sweep_mi300x/dqsweep/A/`；run `20261017-005338-m10-dqs-A-llama-rerun`、`20261017-001322-m10-dqs-A-qwen`）的 k0／k1／k2／K\*；
- stage 40 bench（`parts/dequant_fast_bench.json`）的 torch／fused／compile；
- 3090 的 torch.compile、只載入；
- 編碼時間 `enc_mi300x_*` 與單一 block 的 H2D `h2d_mi300x_*`（parts）。

### 3.2 為什麼有好幾欄：同一個「每 block 反量化」，量法不同差很多

以 Llama `int4_kivi` 的 K0 為例：

| 量法 | 閒 | 忙 |
|:--|--:|--:|
| 單獨量（parts，真實 KV） | 0.496 | 0.637 |
| A 段微測（合成資料） | 0.467 | 0.545 |
| stage 40 bench | 0.449 | NM |
| 還原中・只載入 | 0.448 | 1.701 |
| 還原中・Cake | **0.712** | **1.816** |

- 差別在 Cake 時，載入線的反量化和重算線**同時用 GPU**（09 §5、11 問題一 §2）。GPU 忙時，還原中比單獨量大 2.7–2.9 倍。
- **給 04 的模擬器用「還原中」那一欄**，Cake 用 Cake 欄、全載入用只載入欄。單獨量的欄只能當「kernel 本身多快」。
- 單獨量在 GPU 忙時不一定比閒時慢。例：Qwen `int4_kivi` 單獨 K0 是閒 0.486、忙 0.375；`int2_kivi` 是閒 0.592、忙 0.463。A 段裡也有一格 Qwen `fp8_token` k1 從閒 0.071 跳到忙 0.397。所以**單獨量的「忙」不能拿來當忙時的懲罰倍數**〔判讀〕。

### 3.3 推算、補值、讀取的出處（哪些不是直接量的）

- **沒有推算任何時間**。每一格都是某個 run 的中位數。沒有的就是 NM／P，**沒有用其他格式或其他 GPU 狀態補**。
- stage 43 的還原格子（int2_kivi、mxfp4、K4V2、K8V4 的 K0 還原中）**只在 git ref `origin/main`**（本地 ref，最後一次 fetch 是 commit `1ddba0b`，10/17 13:44Z）。我用 `git show origin/main:results/m10_sweep_mi300x/confirm/speed_<m>/restores.csv.gz` 讀，**沒有切分支**。那 21 份 gpu_guard 都 CLEAN（逐份檢查過）。origin/main 的 RUNLOG 11.2o 寫「43 的速度判讀等品質做完一起寫」。這裡只取反量化時間，**不做任何速度判讀**。
- stage 40 bench 的 JSON 裡**沒有 run_id**。run_id `20261016-130602-m10-dqfast-bench` 是從 RUNLOG_MI300X 11.2f、11_issues 問題一 §4 抄的，那支自測也沒開 gpu_guard。41 海選的 `agg_<m>.csv.gz` 也**沒有 run_id 欄**，run_id 是從 `tables_<m>.md` 的表頭取的。這兩個來源都不完全符合 CLAUDE.md §4.3，表裡照實標。
- 「每 block」＝median(t_dequant ÷ 讀到的低精度 block 數)，取的是比值的中位數。X5 也是這個定義。X7 用的是「中位數的比」；兩種算法在 3090 的 8 格完全相同（§5）。

## 4. 品質（32K）

### 4.1 指標與判準

- **生成**（`q_gen_*`）：對 E0 的配對答對率差，以題為單位 bootstrap 10,000 次的 95% CI。T2–T5、兩輪，1,140 列、600 題。四種情境：A＝全載入；B＝Cake b32，分閒、忙、閒→忙。
  - **過 2 分界線**＝CI 下界 > −2 個百分點（`q_gen_<情境>_pass2`）。
  - `q_gen_eps_pass_all4` 是第二階段的完整判準：四種情境都要 CI 下界 > −2，而且每個任務族、每一輪的點估計 ≥ −5。
  - 出處：`results/m10_precision_mi300x/analysis/{fp8,int4}/quality_<m>_*.json`。Llama 的 FP8-K0／INT8 出自 `results/p3_formats/derived/`。
- **補算的兩格**：用凍結的 `code/m10_analyze.py quality`（一行沒改，subprocess 呼叫），在 Step 1 的同一批列上算（run `20261015-211144`…`20261016-015830`，7 份）。先做兩個重現檢查：Qwen `quality_qwen_step2b.json`、Llama `quality_llama_step1.json` 的 E1。兩個都**逐欄相同**才寫出。
- **teacher forcing**（`q_tf_*`）：41 海選的 ALL（block 1–63 全部低精度），96 題、只有第 1 輪。
  - `first_ok`＝第一個答案 token 對不對，是代理指標，**不是生成答對率**。E0 的 first_ok：Llama 42.7%、Qwen 18.8%。
  - 只有點估計：逐題的 CSV 在 MI300X 的 `/mlsteam/data/tiara/sweep_m10/`，不在 repo，所以算不了 CI。
  - `q_tf_A_pass2_point`＝點估計 > −2（09 §4.2-3 的前緣點規則），**不是登記的品質判準**。
- **格式小測**（`q_fmtsel_kl_A`）：凍結前 20 題的 E1 讀法 A 平均 KL，`fmt_select/fmt_choice.json`。
- 生成的 KL（`q_gen_mean_kl_A`，在生成的 token 上算）和 teacher forcing 的 KL（在標準答案的位置上算、題目也不同）**定義不同，不能互比**。例：Llama INT4 是 0.0068 和 0.0042。

### 4.2 表

**Llama**

| 格式 | 判定（32K） | 生成：全載入 差 [95% CI]（pt） | Cake 閒／忙／閒→忙 的 CI 下界 | 四種情境都過 | 生成 KL（A） | TF first_ok 差 A（pt） | TF KL（A） | TF 點估計 > −2 | 待補 stage |
|:--|:--|:--|:--|:--|--:|--:|--:|:--|:--|
| bf16 | 基準 | — | — | — | — | — | — | — | — |
| fp8_token | **PASS** | −0.47 [−1.14, 0.18] | −0.78／−1.08／−0.64 | 是 | 0.003488 | 0.0 | 0.002965 | 是 | — |
| fp8e5_head | 只有 TF | P | NM（44c 只用讀法 A） | NM | P | +2.08 | 0.032398 | 是 | 44c |
| fp8k0_token | **PASS**（補算） | +0.15 [−0.52, 0.82] | −0.56／−0.10／−0.61 | 是 | 0.001829 | 0.0 | 0.001851 | 是 | — |
| int8_token | **PASS**（補算） | −0.12 [−0.68, 0.41] | −0.22／−0.03／−0.27 | 是 | 0.000528 | 0.0 | 0.000429 | 是 | — |
| kv-int8_token-int4_kivi | 只有 TF | P | NM | NM | P | −1.04 | 0.00105 | 是 | 44c |
| int4_kivi | **PASS** | +1.09 [0.31, 1.87] | −0.35／−0.31／−0.57 | 是 | 0.006821 | 0.0 | 0.004235 | 是 | — |
| int4_token | 只有 TF | P | NM | NM | P | −2.08 | 0.049623 | **否** | 44c |
| int4h_token | 只有 TF | P | NM | NM | P | 0.0 | 0.014872 | 是 | 44c |
| mxfp4 | 只有 TF | P | P | P | P | 0.0 | 0.067425 | 是 | 43、44c |
| nvfp4 | 只有 TF | P | NM | NM | P | 0.0 | 0.050267 | 是 | 44c |
| kv-int4_kivi-int2_kivi | 只有 TF | P | P | P | P | +1.04 | 0.018343 | 是 | 43、44c |
| **int2_kivi** | 只有 TF | P | P | P | P | −1.04 | 0.136954 | 是 | **43**、44c |
| int2h_g32 | 只有 TF | P | NM | NM | P | −2.08 | 0.301334 | **否** | 44c |
| fp8_head（另一種粒度） | NM | NM | NM | NM | NM | NM | NM（格式小測 KL 0.004342） | NM | 不會補 |

**Qwen**

| 格式 | 判定（32K） | 生成：全載入 差 [95% CI]（pt） | Cake 閒／忙／閒→忙 的 CI 下界 | 四種情境都過 | 生成 KL（A） | TF first_ok 差 A（pt） | TF KL（A） | TF 點估計 > −2 | 待補 stage |
|:--|:--|:--|:--|:--|--:|--:|--:|:--|:--|
| bf16 | 基準 | — | — | — | — | — | — | — | — |
| fp8_head | **FAIL** | −32.84 [−35.75, −29.97] | −28.47／−30.71／−29.46 | 否 | 3.83809 | −5.21 | 3.748226 | 否 | — |
| fp8e5_head | 只有 TF | P | NM | NM | P | −17.71 | 8.42975 | 否 | 44c |
| fp8k0_head | **PASS** | +0.39 [−0.32, 1.08] | −0.53／−0.36／−0.65 | 是 | 0.004874 | 0.0 | 0.004904 | 是 | — |
| int8_token | **PASS** | +0.20 [−0.48, 0.87] | −0.41／−1.00／−0.68 | 是 | 0.008262 | 0.0 | 0.005215 | 是 | — |
| kv-int8_token-int4_kivi | 只有 TF | P | P | P | P | 0.0 | 0.004471 | 是 | 43、44c |
| int4_kivi | **PASS** | +0.06 [−0.78, 0.88] | −0.01／−0.30／−0.07 | 是 | 0.013368 | 0.0 | 0.006331 | 是 | — |
| int4_token | 只有 TF | P | NM | NM | P | −18.75 | 10.804021 | 否 | 44c |
| int4h_token | 只有 TF | P | NM | NM | P | −18.75 | 11.766803 | 否 | 44c |
| mxfp4 | 只有 TF | P | NM | NM | P | −18.75 | 11.024907 | 否 | 44c |
| nvfp4 | 只有 TF | P | NM | NM | P | −12.5 | 13.927845 | 否 | 44c |
| kv-int4_kivi-int2_kivi | 只有 TF | P | P | P | P | −1.04 | 0.051644 | 是 | 43、44c |
| **int2_kivi** | 只有 TF | P | NM | NM | P | **−6.25** | 0.228312 | **否** | **只有 44c** |
| int2h_g32 | 只有 TF | P | NM | NM | P | −16.67 | 13.409718 | 否 | 44c |
| fp8_token（另一種粒度） | NM | NM | NM | NM | NM | NM | NM（格式小測 KL 11.78） | NM | 不會補 |

- Qwen 的 E0 first_ok 只有 18.8%，地板很低，first_ok 低估了傷害。例：Qwen FP8 的 TF 是 −5.2，生成是 −32.8。**只有 TF 的格子不要當成「過」**〔判讀〕。
- 對 04 §3 的意思：
  - O 要用的 INT2，**Llama** 的生成品質會在 stage 43 出來（`E1s@int2_kivi`）。
  - **Qwen** 的 INT2 KIVI **不在 43 的配置裡**（`confirm_plan.json` 的 Qwen E1s 只有 K4V2、INT4 KIVI、K8V4），要等 44c。44c 只用讀法 A、題數比主線少。
  - Qwen INT2 KIVI 的 TF 點估計已經掉 6.25 分。照 04 §3，在生成品質出來前只能給「不會再被讀」的 session 用；以目前的 TF 數字看，Qwen 的 INT2 很可能不過〔判讀，不是判定〕。

## 5. 對拍與矛盾（兩邊都列，寫出差在哪）

**一致的**（`format_table.json` 的 `crosschecks`）：
- X5 `x5_cost_constants.json`（run `20261017-215824-p3x5-loadonly`）的 INT4 KIVI 還原中每 block 反量化，兩個模型 × K0／融合 × 閒／忙 8 格，和本表**完全相同**（例：Llama K0 閒 0.7122、忙 1.8163；Qwen 融合忙 0.3505）。BF16／INT4 每 block 位元組也相同。
- X7 `constants_3090.json` 的 Qwen 還原中每 block 反量化 8 格，和本表從原始 restores 重算的**完全相同**。
- 補算的生成品質：兩個重現檢查逐欄相同（§4.1）。
- 本表的品質數字和 08 §2.3–§2.5、main.tex `tab:quality` 一致。例：Qwen FP8 A −32.8 [−35.8, −30.0]、Llama INT4 A +1.1 [+0.3, +1.9]。

**有差的**：

| # | 甲 | 乙 | 差在哪 | 處理 |
|:--|:--|:--|:--|:--|
| 1 | X7 §4.3：「MI300X 沒有存單獨的 bench 數字 → NOT_MEASURED」 | `results/m10_sweep_mi300x/parts/dequant_fast_bench.json`（stage 40，和 3090 同一支 `--bench`）**有**，閒：Llama fp8_token 0.2272、int4_kivi 0.4488；Qwen fp8_head 0.099、int4_kivi 0.2544 ms | X7 只找了 `dqfast_selftest_gpu.json`（只有 checks），漏了這份。X7 拿 3090 的單獨量去比 MI300X 的「還原中・只載」，得到「主線寫法慢 2–4 倍」。同一支工具比的話：**K0 慢 3.6–5.1 倍、融合慢 1.7–2.6 倍**（10 格，見 CSV） | 兩個都列；本表的 3090 對 MI300X 用同一支 bench |
| 2 | Qwen `int4_kivi` K0 單獨量：stage 40 bench 0.2544、還原中只載入 0.2345 ms | parts 0.4859、A 段 0.4802 ms（閒） | 同一個量差 1.9 倍。Llama 這三種量法都在 0.45–0.50。原因 NOT_MEASURED（〔推論〕parts／A 段寫進非連續視圖，bench 不是；沒查證） | 兩個都列；模擬器用還原中那欄 |
| 3 | stage 43 的 `int4_kivi` E1s Cake CPU 層每 block K0（閒）：Llama 0.7846、Qwen 0.4589 | 主線 INT4 格子同一格：0.7145、0.4170 | 同格式、同寫法、不同 run 差 +9.8％／+10.0％；忙時只差 +0.4％／−0.3％ | 兩個都列；43 的新格式（int2 等）的閒時數字可能也有約 10% 的 run 間差〔推論〕 |
| 4 | X7 用 Qwen FP8 的第一次量 `20261015-175050-m10-grid-fp8-qwen`（L0 監控報過污染，32K 的列沒被標） | 第二階段判定用重量 `20261015-194150-m10-regrid-fp8-qwen` | 還原中每 block：Cake 閒 0.1712 vs 0.1725、忙 0.4229 vs 0.4147；只載 0.1052 vs 0.1058、0.4059 vs 0.4059 | 本表用重量那次 |
| 5 | 08 §2.8 A1 文字：「每個 block，Llama … FP8 32 MiB」 | 實際 34,603,008 B＝33.0 MiB（含每 token 的 FP32 scale）；`iowall.json` 用的是正確位元組 | 文字四捨五入，差 3.1% | 本表用實際值；08 不改（不准改既有檔） |
| 6 | 09 §2.1 的位元／值：FP8-K0 8.27／8.02 | 本表 Llama 8.2656、Qwen 8.0161 | 只是取到小數 2 位 | 無 |

## 6. 缺什麼（NOT_MEASURED，而且沒有排程會補的）

- **3090**：
  - GPU 忙時的單獨反量化：X7 的 bench 只在閒時跑。
  - Llama 的還原中：X7 兩次都被別人的行程污染。
  - 任何格式的融合版還原中：X7 的 `grid_fused`＝[]。
  - 8 種新格式（int2、mxfp4、nvfp4、Hadamard、K／V 分開、e5m2、FP8-K0）的任何計時。
  - torch.compile 的 FP8 在 sm_86 上編不起來（Triton 不支援 `fp8e4nv`，X7 §4.3），所以 3090 沒有 FP8 的 compile 數字。
- **MI300X 融合 kernel**：
  - fp8e5、fp8k0、int4h、nvfp4、int2h 從來沒有融合版。
  - Qwen 的 int2_kivi、mxfp4，以及 Llama 的 K8V4，**不在 43 的速度格子裡**，所以 43d 也不會量。
- **MI300X 還原中（K0）**：fp8e5、int4_token、int4h、nvfp4、int2h，以及 Qwen 的 int2_kivi、mxfp4，沒有任何還原格子。只有單獨量。
- **MI300X 只載入（K0）**：2b 與 43 的格子都沒跑只載的參考線，所以 FP8-K0、INT8 和新格式都沒有。
- **生成品質**：「另一種 FP8 粒度」的兩列（只有格式小測的 KL）；teacher forcing 的 CI。
- **反量化與長度**：只有 32K。

## 7. 待補（PENDING；`quality_pending.csv`）

| stage | 補什麼 | 涵蓋的均勻格式 |
|:--|:--|:--|
| 43（43ab） | 32K 生成答對率（新題 seed 50000 起，T2–T4 各 90、T5 30、兩輪；讀法 A＋B b32） | Llama：int2_kivi、K4V2、int4_kivi、mxfp4；Qwen：K4V2、int4_kivi、K8V4 |
| 43d | 新格式融合 kernel 的 CPU 層還原時間（13 號 Q3） | Llama：int2_kivi、mxfp4、K4V2；Qwen：K4V2、K8V4 |
| 44a | 64K／128K 速度格子（原寫法） | FP8 主線粒度、int4_kivi |
| 44b → 44c | 32K／64K／128K 的 teacher forcing＋每個格式 ALL 的生成（讀法 A） | 13 種全部（Qwen INT2 KIVI 的生成品質只會在這裡出現） |
| 44d | 長 context 的還原第二輪（13 號 Q1） | FP8 主線粒度、int4_kivi |

## 8. 失敗與異常（完整）

1. **`20261018-023324-p3fmt-table` exit 1**，這是我自己的程式錯：
   ```
   File "/home/hungwei/paper-hierarchical-kv-state/code/p3_formats/build_format_table.py", line 285, in main
       dq3 = dq3[dq3.shape.astype(str) == f"{m}_block"]
   AttributeError: 'tuple' object has no attribute 'astype'
   ```
   原因：`shape` 是 pandas DataFrame 的屬性，不是欄位。改成 `dq3["shape"]` 後重跑 `20261018-023339`（exit 0）。這不是換方法：同一支程式只修欄位名稱，資料與算法都沒變。
2. `20261018-023339` 之後，我改了「PENDING」的標法：
   - 43d 只量 43 速度格子裡的格式，原本把 Qwen int2_kivi／mxfp4 也標成 PENDING，是錯的，改成 NOT_MEASURED。
   - 「另一種 FP8 粒度」不在 44c 裡，改成 NOT_MEASURED。
   - `pending_stages` 原本被冒號截斷，改用 stage 編號。
   - BF16 的融合欄原本是空的，補成 0。

   改完重跑 `20261018-023443`（這一版）。數值欄都沒變。
3. 沒有補 RUNLOG：任務規則是「不要改既有檔案」，所以這次的記錄都在本頁與 run 目錄裡，請主 session 決定要不要補一段到 `results/RUNLOG_3090_P3.md`。
4. 此次整理為 self-review，非 cross-model。
