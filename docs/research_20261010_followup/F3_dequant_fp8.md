# F3 反量化成本（D6 收尾）＋ 九月的 FP8 KV 有沒有生效

> **agent**：F3（後續 5 個 agent 之一）。只建新檔，不改既有檔，不改 main.tex，不 commit。
> **判準登記**：2026-10-10T07:40:48Z（`date -u`），寫在任何量測、任何舊 log 查看之前。§2 是登記原文，之後不改。
> **標記**：〔實測 run_id〕〔算術〕〔程式碼 file:line〕〔判讀〕〔未查證〕。
> **閱讀順序**：§2（判準）放在最上面，因為它是量測前寫的、之後沒改過的原文。結論直接看 §0。
> **run**：`20261010-074630-f3-dequant`（工作 A）、`20261010-074830-f3-fp8`（工作 B）。GPU 佔用合計約 11 分鐘（A 約 0.5 分、B 約 10.3 分）。本文完成於 2026-10-10T08:11:56Z 之後。

## 2. 事先寫好的判準（pre-registration，2026-10-10T07:40:48Z，量測前）

### 2.A 工作 A：反量化一個 64 MiB chunk 要多久

**D6 的判準原文（照抄 `docs/research_20261010_directions/D6_precision.md` §6 第 0 步）：**

> 判準（現在寫死）：INT8 反量化 ≤1 ms／chunk → 依 §2.4 判死路（帶內 96 格全部 0.0%，§4.2）；≥3.5 ms 且無法和重算並行 → 維持「可能」，但結論改成「高頻寬時混入 BF16 分擔反量化」，這和位置無關。

**我怎麼套用（登記時寫死）：**
- 量的東西：Llama-3.1-8B 一個 chunk ＝ 512 token × 32 層 × K/V × 8 head × 128 dim ＝ 33,554,432 個值，BF16 是 64 MiB。
- 判定用的數字：**INT8→BF16、每 token 每 head 一個 scale（和 D6 §2.1 的位元組模型一致）**，在「最快、且輸出正確」的實作上，每 chunk 的**中位數**。
  - 「正確」＝和 FP32 參考算法比，最大誤差 ≤ BF16 的捨入誤差（相對誤差 ≤ 2^-7）。不正確的實作不算。
  - 另外報 per-group（group=128，即每 token 每 head 一組，和上面同形狀；再加 group=32）、FP8 e4m3fnuz→BF16、INT4→BF16（unpack＋scale）。
  - 每個實作：warmup ≥5 次，量 ≥20 次（用 HIP event），報中位數和 p90。
- 判定（照 D6 原文）：
  - INT8 中位數 **≤1 ms** → D6 **死路**。
  - INT8 中位數 **≥3.5 ms**，而且無法和重算並行 → 維持「可能」，結論改成「高頻寬時混入 BF16 分擔反量化，和位置無關」。
  - **1–3.5 ms**（D6 原文沒寫）：我事先決定查 D6 §4.2 的門檻細掃表（Q1：3.0 ms 時 3.9%，3.5 ms 時 11.7%）。若 Q1 在量到的值 <5% → 死路；否則「可能」。這條是我加的，標為〔判讀〕。
- 補充（不進判定）：INT4 若 >1.5 ms，要指出 D6 的 Q2 欄可能受影響（Q2 在 2.0 ms 時 9.4%）。
- 和 D6 引用的舊值（INT4 0.814、FP8 0.087 ms／chunk，run 20260915-124058）比較，並查清楚舊值是怎麼量的。

### 2.B 工作 B：九月品質 run 的 FP8 KV 有沒有真的生效

**判定規則（照任務書，寫死）：**
- **生效**：KV 容量 fp8 ÷ auto ≈ 2×（我定為 1.8–2.2×），**而且** logprob 有差。
- **沒生效**：容量沒變（0.95–1.05×），**或** 所有 prompt 的所有記錄 token 的 logprob 都位元相同。
- **不確定**：其他情況。

**細節（寫死）：**
- 「logprob 有差」＝至少一個 prompt 的前 N 個生成 token（N=8）中，fp8 的 top-1 logprob 和 auto 的不同（位元比較）。
- 雜訊對照：auto 跑兩次（A1、A2）。如果 A1、A2 本身就不同，那 fp8 的差要**大於** A1–A2 的最大差才算「有差」；否則判「不確定」。
- 版本：用九月那幾個 run 實際用的 vLLM（從 cmd.sh／context.txt 查），同一個模型，greedy（temperature=0）。prompt 3–5 個、長度 32K–64K。
- 也看 vLLM 的 log：有沒有說用 fp8 存 KV、選了哪個 attention backend、KV 容量（GPU blocks／tokens）。backend 的選擇要引 vLLM 原始碼 file:line。
- 九月 run 的判定：先看九月 log（傳了什麼 kv_cache_dtype、容量）；若 log 不足，就用同版本重現的結果推論，並標〔判讀〕。

### 2.C 補登（2026-10-10T07:43:50Z，仍在任何 GPU 量測之前；只補細節，不改上面的規則）

- **快取冷熱**：MI300X 有 256 MB 的 Infinity Cache。一個 chunk 的輸入＋輸出約 96 MiB，放得進去，反覆用同一塊 buffer 會量到「快取熱」的偏快數字。
  - 所以每個實作量兩種：**cold**（輪流用 8 組不同的 buffer，總量 >700 MiB，超過快取）和 **hot**（同一組 buffer）。
  - **判定用 cold 的中位數**（比較保守）。hot 只報告。
- **工作 B 的 prompt**：自己產生撈針式的長 prompt（每個開頭不同，避免 prefix cache 互相命中），長度約 32K、48K、64K、64K，外加 1 個約 120K（對照九月的 129K）。判定只看 32K–64K 那 4 個；120K 那個只報告。
- **工作 B 的 vLLM 版本**：九月的 server 是哪個 venv，從 server.log 查（見 §3）；重現用同一個。`--max-model-len 129536`、`--gpu-memory-utilization 0.90`、其他用預設（和九月一樣）。每個 prompt 生成 32 個 token，記前 8 個的 top-1 logprob 做判定。
- **組別**：auto（A1）、auto（A2）、fp8（F1）、fp8_e4m3（F2，確認是不是 fp8 的別名）。

## 0. 一句話結論（A 和 B 各一個判定）

**A（D6 收尾）：死路。**
- INT8→BF16 反量化一個 64 MiB 的 chunk，最快的正確實作只要 **0.031 ms**（中位數；p90 0.033 ms；快取冷）〔實測 20261010-074630-f3-dequant〕。
- D6 的門檻是 1 ms，這個值小了約 32 倍〔算術〕。最慢的 plain torch 寫法也只要 0.184 ms。
- 照 D6 §6 的判準原文：INT8 ≤1 ms → D6 判**死路**。

**B（九月的 FP8 KV 有沒有生效）：生效。**
- 九月兩個 run 的 log：有傳 `--kv-cache-dtype fp8`，vLLM 也印出「Using fp8 data type to store kv cache」。KV 容量從 1,267,504 變成 2,535,008 token，剛好 2.000 倍，用的記憶體一樣是 154.73 GiB〔實測 20260916-062139、20260917-103133 的 server.log〕。
- 用同一個 vLLM（0.28.0）重現：容量一樣是 2.000 倍。4 個 32K–64K 的 prompt 裡，FP8 的 logprob **4 個都和 BF16 不同**；BF16 自己跑兩次則**位元完全相同**〔實測 20261010-074830-f3-fp8〕。
- 但 4 個裡有 3 個輸出文字和 BF16 逐字相同。所以九月「129K 時 20/20 逐字相同」不代表 FP8 沒開：答案很短，greedy 下模型很確定，小誤差改不了選中的字〔判讀〕。

**三個關鍵數字**

| # | 數字 | 意思 | 標記 |
|:--|:--|:--|:--|
| 1 | 反量化一個 64 MiB chunk（cold，最快的正確實作）：INT8 **0.031 ms**、FP8 0.026 ms、INT4 0.027 ms | 全部比 D6 的 1 ms 門檻小 30 倍以上 | 〔實測 20261010-074630-f3-dequant〕 |
| 2 | D6 引用的「INT4 0.814、FP8 0.087 ms／chunk」**不是反量化 kernel 的時間** | 那是 warm TTFT 的差。每組只有 3 個樣本，INT4 那組還換了 attention kernel（TRITON_ATTN 對上 ROCM_ATTN） | 〔實測 20260915-124058 的 stdout 與 server.log〕〔判讀〕 |
| 3 | FP8 容量 **2.000×**；logprob **4/4** 不同；BF16 重跑差 **0** | FP8 真的存成 fp8，而且確實改變了計算 | 〔實測 20261010-074830-f3-fp8〕 |

## 1. 問題

### 工作 A：反量化到底多貴？

- D6 問的是：「依位置挑精度」能不能比「全部用同一種精度」還原得更快？例如前段 INT8、尾巴 BF16。
- D6 算出來，答案取決於「反量化」有多貴。
  - **反量化**：把壓縮過的數字（INT8／FP8／INT4）轉回 BF16，attention 才能用。每載入一個 chunk 就要做一次。
  - 反量化便宜（≤1 ms／chunk）→ 混精度完全沒好處（D6 §4.2：96 格全部 0.0%）。
  - 反量化貴（≥3.5 ms）→ 混精度才可能贏，但贏的原因和位置無關。
- 問題是 INT8 的反量化成本**沒量過**，所以 D6 只能判「可能」。
- **要證明什麼**：在 MI300X 上反量化一個 64 MiB 的 chunk 要幾 ms。這只是計時，不是有損實驗。

### 工作 B：九月的 FP8 是不是根本沒開？

- D6 盤點時發現一件怪事：Llama-3.1-8B 在 129K 時，FP8 的輸出和 BF16 **20 題全部逐字相同**；65K 時是 17/20。
- 擔心的是：vLLM 會不會靜默地退回 BF16？如果真是這樣，所有「MI300X 上 FP8 沒有損失」的結論都不能用。
- **要證明什麼**：九月那兩個 run 的 KV 是不是真的存成 fp8。
- 白話的分辨方法：
  - **看容量**：fp8 一個數 1 byte，BF16 一個數 2 byte。真的開了，同樣的記憶體就能放 2 倍的 token。
  - **看 logprob**：logprob 是模型對每個字有多確定的分數。KV 有誤差時，就算選到同一個字，分數也會有一點點不同。

## 3. 做了什麼

| 步驟 | 內容 | 程式／run |
|:--|:--|:--|
| A1 | 查 D6 舊值怎麼量的：讀 run `20260915-124058-m2-b-llama8b-retrieval` 的 stdout，以及內層 `20260915-124109-m2-retrieval` 的 18 個 server.log | 只讀 log |
| A2 | 反量化計時：6 種格式 × 2–6 種實作。每種先和 FP32 參考答案比對正確性，再量 cold／hot 各 50 次（先暖身 10 次），用 HIP event | `code/m9_f3_dequant.py`，run `20261010-074630-f3-dequant` |
| B1 | 讀九月兩個 run 的 cmd.sh、stdout，以及內層 run（`20260916-062212-m5-needle`、`20260917-103206-m5-needle`）的 10 個 server.log | 只讀 log |
| B2 | 讀 vLLM 0.28.0 原始碼：ROCm 怎麼選 attention backend、ROCM_ATTN 怎麼處理 fp8 | 見 §4.6 |
| B3 | 同版重現：依序開 4 個 server（auto、auto、fp8、fp8_e4m3），送同一批 5 個 prompt（token id 完全相同），greedy 生成 32 個 token，記錄每個 token 的 logprob | `code/m9_f3_fp8_repro.py`，run `20261010-074830-f3-fp8` |

**共同設定**
- 兩個 GPU 指令都包在 `flock /mlsteam/data/tiara/gpu.lock` 和 `code/m7_guard_run.py` 裡，用 `m7run` 開 run 目錄。程式一開始就 `chdir` 到 run 目錄（避免 gpucore 檔掉在 repo）。
- venv：`/mlsteam/workspace/venv/tiara-v028`（vLLM 0.28.0、torch 2.12.1+rocm7.2、Triton 3.8.0）。這和九月 server 用的是同一個（§4.4）。

**工作 A 的細節**
- 資料形狀：262,144 列 × 128。每一列是一個 (層, K/V, token, head)，總共 33,554,432 個值，BF16 剛好 64 MiB。
- 格式與 scale：
  - INT8：每 (token, head) 一個 FP32 scale（和 D6 的位元組模型一樣：32+1 MiB）；另做每 32 個值一個 scale（g32）。
  - FP8：e4m3fnuz（MI300X 的格式），整個 tensor 一個 scale（＝vLLM 的 `fp8`）；另做每 (token, head) 一個 scale（＝`fp8_per_token_head`）。
  - INT4：一個 byte 放兩個 4-bit 值，每 (token, head) 一個 FP32 scale（對稱，零點 8）；另做 vLLM `int4_per_token_head` 的 scale 格式（零點放在 scale 的低 4 bit）。**沒有做 vLLM INT4 的 Hadamard 旋轉（RHT）**。
- 實作：plain torch（2–4 種寫法）、`torch.compile`、自己寫的 Triton kernel、vLLM 的 `convert_fp8`（vLLM 唯一的獨立反量化 kernel，只支援 FP8；原始碼註明「Only for testing」，`csrc/libtorch_stable/cache_kernels.cu:954-1029`）。vLLM 的 INT8／INT4 反量化都融合在 attention kernel 裡，沒有獨立的 kernel。
- **cold**：8 組 buffer 輪流用（約 776 MiB），超過 MI300X 的 256 MB Infinity Cache。**hot**：同一組 buffer。判定用 cold。
- 每次量之前先讓 GPU `torch.cuda._sleep` 一下，讓 CPU 先把 kernel 排好。所以量到的是 GPU 上的執行時間，不含 CPU 發 kernel 的空檔。

**工作 B 的細節**
- Prompt：自己產生的撈針式 prompt（每個開頭都不同，避免 prefix cache 互相命中）。長度 32,768、49,152、65,536、65,536（判定用），外加 122,880（只報告）。
- Server 參數照九月：`--max-model-len 129536 --gpu-memory-utilization 0.90`，其他用預設。請求：temperature 0、seed 12345、32 個 token、`logprobs=1`、回傳 token id。
- 每個 server 開之前，GPU 都是空的（log：Free memory 191.46/191.98 GiB）。

## 4. 結果

### 4.1 工作 A：反量化一個 64 MiB chunk 要多久（MI300X）

來源：`results/m9_followup/f3_dequant.csv`〔實測 20261010-074630-f3-dequant〕。全部是 cold、50 次；括號是 p90。

| 格式 | 最快的正確實作 | 次快 | plain torch 最快寫法 | plain torch 最慢寫法 |
|:--|:--|:--|:--|:--|
| **INT8，每 (token, head) 一個 scale**（判定用） | **torch.compile 0.0309（0.0327）** | Triton 0.0317（0.0327） | `torch.mul(q, s, out=)` 0.0956（0.0964） | 轉 FP32 再乘 0.1843（0.1879） |
| INT8，每 32 個值一個 scale | torch.compile 0.0274（0.0288） | Triton 0.0306（0.0322） | `torch.mul(out=)` 0.0965（0.0969） | 0.1854（0.1886） |
| FP8 e4m3fnuz，整個 tensor 一個 scale | torch.compile 0.0261（0.0268） | vLLM `convert_fp8` 0.0495（0.0518） | `.to(bf16)` 0.0809（0.0819） | 0.1892（0.1924） |
| FP8，每 (token, head) 一個 scale | bf16 乘法 0.1252（0.1271） | — | 同左 | 0.1956（0.1988） |
| INT4（對稱） | Triton 0.0274（0.0278） | torch.compile 0.0313（0.0317） | bf16 寫法 0.1778（0.1792） | 0.2741（0.2794） |
| INT4（vLLM 的 scale 格式，無 RHT） | Triton 0.0276（0.0279） | — | — | — |

單位 ms／chunk。

**看到什麼**
- **全部都遠低於 1 ms。** 連最笨的 plain torch 寫法也 ≤0.28 ms。
- 每 token 換算：INT8 0.060、FP8 0.051、INT4 0.053 µs/token（cold 最快 ÷ 512）〔算術〕。
- 已經接近記憶體頻寬的極限：讀 33 MiB、寫 64 MiB，以 MI300X 峰值 5.3 TB/s 算，下限約 0.019 ms〔算術〕。INT8 最快的那個跑到 3.3 TB/s，是峰值的 62%〔算術〕。
- hot（同一組 buffer）快 0–40%（例：INT8 Triton 0.0317 → 0.0259 ms；變化最大的是 vLLM convert_fp8 0.0495 → 0.0307 ms）。不管冷熱都遠低於 1 ms。
- 每個成功的實作都通過正確性檢查：最大相對誤差 ≤7.5×10⁻³，門檻是 2⁻⁷ ≈ 7.8×10⁻³。
- FP8 per-token-head 只有 plain torch 的數字，因為我寫的 Triton 版本編譯失敗（§6）。這個格式不進判定。

### 4.2 工作 A：套 D6 的判準

| 條件（§2.A） | 結果 | 依據 |
|:--|:--|:--|
| INT8 中位數 ≤1 ms → 死路 | **成立**：0.0309 ms（p90 0.0327） | 〔實測 20261010-074630-f3-dequant〕 |
| 1–3.5 ms 的補充規則 | 用不到 | — |
| 補充：INT4 >1.5 ms 會影響 Q2 欄嗎 | 不會：INT4 0.027 ms | 同上 |

- D6 §4.2 的表裡，d=0.1 ms 那欄的 Q1、Q2 都是 **0.0%**（帶內 96 格，反量化放 GPU 線）〔算術 20261010-055349-d6-arith〕。量到的最快值全部 <0.1 ms，所以混精度在 D6 的模型裡沒有任何增益。
- 就算用最慢的 plain torch（≤0.28 ms），也落在 d=0.5 ms 那欄裡面，同樣全部 0.0%〔算術 20261010-055349-d6-arith〕。
- D6 §2.4「死路」還要求延後版能到達同樣的最終狀態。這一點 D6 §4.6 已經判為成立，我沒有重做。
- **判定：死路。**
- 沒量的：「反量化和重算同時跑時，重算會變慢多少」（D6 §6 第 0 步的第二部分）→ **NOT_MEASURED**。判讀：就算爭用讓反量化慢 10 倍，也才約 0.3 ms，仍然 <1 ms〔判讀〕。
- 我沒有重跑 `code/m8_precision_arith.py`。它會覆寫 `results/m8_directions/` 裡的既有檔，違反「只建新檔」的規則。D6 現有的敏感度表已經涵蓋這個範圍。

### 4.3 工作 A：和 D6 引用的舊值比

| 精度 | D6 引用（每 chunk） | 這次：獨立反量化，cold 最快 | 差幾倍〔算術〕 |
|:--|:--|:--|:--|
| INT4 | 0.814 ms | 0.027 ms（Triton） | 約 30 倍 |
| FP8 | 0.087 ms | 0.026 ms（torch.compile）／0.049 ms（vLLM convert_fp8） | 1.8–3.3 倍 |
| INT8 | NOT_MEASURED | 0.031 ms | — |

**舊值是怎麼量的**（讀 run `20260915-124058` 的 stdout 與內層 `20260915-124109-m2-retrieval` 的 server.log）
- 量的是 **warm TTFT**。同一個 16,384-token 的 prompt 送第二次（KV 已經在 GPU 上），記錄到第一個 token 出來的時間。減掉 BF16 的 warm TTFT，再除以 32 個 chunk〔實測 20260915-124058〕〔算術〕。
- 每個設定只有 **3 個樣本**（3 輪各 1 次），取中位數：
  - BF16：85.2、81.7、78.2 ms；
  - FP8：74.6、84.5、88.8 ms；
  - INT4：107.8、103.3、110.3 ms。
  - FP8 的 +2.7 ms，比 BF16 自己的跳動（7 ms）還小，所以在雜訊內。
- **INT4 那組換了 attention kernel。** BF16、FP8 用 ROCM_ATTN；INT4（`int4_per_token_head`）用 TRITON_ATTN〔實測 20260915-124109-m2-retrieval/*/server.log〕。所以 +26.1 ms 混了兩件事：「換一個 attention kernel」和「attention 讀 cache 時順便反量化」。這不是獨立反量化的時間。
- D6 自己在 §7 也寫過：舊值是「attention 讀量化 cache 的額外時間」，和「載入後一次性反量化」可能不同。這次量的是後者，也就是 D6 成本模型裡 d 的定義。

### 4.4 工作 B：九月的 log 怎麼說

**用的是哪個 vLLM**
- cmd.sh 是用 `venv/tiara`（vLLM 0.19.1）的 python 跑 `code/m5_quality.py`。
- 但 m5_quality.py 起 server 時用的是 `/mlsteam/workspace/venv/tiara-v028/bin/vllm`（每個設定的 cmd.txt）。server.log 印的是 `version 0.28.0`〔實測 兩個內層 run 的 server.log〕。
- **注意**：九月 run 的 context.txt 記的「vllm commit b1388b1f…」是 0.19.1 那份原始碼（`/mlsteam/workspace/src/vllm`）的 commit，**不是實際跑的 server**。0.28.0 的原始碼是 `src/vllm-v0.28.0`，commit 2cf0a691。這很容易誤導。
- 程式原碼：`code/m5_quality.py` 已在 commit 52803bc 從 `code/` 刪掉。我讀的是 git 裡 e2ff39d 的版本（`VENV` 預設 `tiara-v028`）。

**九月兩個 run 的各設定**（65K 和 129K 兩個 run 的數字完全一樣）

| 設定 | `--kv-cache-dtype` | attention backend | KV 容量（token） | 可用 KV 記憶體 | 對 BF16〔算術〕 |
|:--|:--|:--|--:|--:|--:|
| bf16 | 沒傳（auto） | ROCM_ATTN | 1,267,504 | 154.73 GiB | 1.000× |
| **fp8** | fp8 | **ROCM_ATTN** | **2,535,008** | 154.73 GiB | **2.000×** |
| fp8_ptk | fp8_per_token_head | TRITON_ATTN | 2,458,192 | 154.73 GiB | 1.939× |
| int8 | int8_per_token_head | TRITON_ATTN | 2,458,192 | 154.73 GiB | 1.939× |
| int4 | int4_per_token_head | TRITON_ATTN | 4,771,792 | 154.73 GiB | 3.765× |

- fp8 的 server.log 有「Using fp8 data type to store kv cache」（來自 `vllm/config/cache.py:282-283`），engine config 也印出 `kv_cache_dtype=fp8`〔實測〕。
- 算術對照：BF16 每個 token 的 KV 是 128 KiB（32 層 × K/V × 8 head × 128 × 2 byte）。154.73 GiB ÷ 128 KiB ≈ 1,267,548，和 log 的 1,267,504 對得上（差在 block 取整與 GiB 顯示的四捨五入）〔算術〕。fp8 每 token 64 KiB → 2 倍。
- **結論**：九月的 log 已經直接顯示 fp8 有被用來存 KV。

### 4.5 工作 B：同版重現

來源：`results/m9_followup/f3_fp8_repro_gens.csv`、`f3_fp8_repro_compare.csv`〔實測 20261010-074830-f3-fp8〕。

- vLLM 0.28.0（`tiara-v028`；原始碼 commit 2cf0a691，`git status` 乾淨；`.so` 建於 2026-09-15；沒有任何 .py 比九月的 run 新）→ 和九月是同一份 build。

**容量**

| 組 | `--kv-cache-dtype` | backend | KV 容量（token） | 可用 KV 記憶體 | 對 A1〔算術〕 |
|:--|:--|:--|--:|--:|--:|
| A1 | auto | ROCM_ATTN | 1,271,056 | 155.16 GiB | 1 |
| A2 | auto | ROCM_ATTN | 1,271,056 | 155.16 GiB | 1.000× |
| F1 | fp8 | ROCM_ATTN | 2,542,080 | 155.16 GiB | **2.000×** |
| F2 | fp8_e4m3 | ROCM_ATTN | 2,535,008 | 154.73 GiB | 1.994×（每 GiB 換算 2.000×） |

- F2 少了 0.43 GiB 的可用記憶體，是那次 server 啟動時量到的可用量不同，不是格式不同。換算成每 GiB：16,383 對 8,192 token，仍是 2.000 倍〔算術〕。

**logprob**（F1 對 A1；前 8 個生成 token）

| prompt | 長度 | 針的深度 | 找到針（A1／F1） | 文字和 A1 逐字相同？ | logprob 位元相同？ | 最大 \|Δlogprob\| | 第 1 個 token 的 \|Δ\| |
|:--|--:|--:|:--|:--|:--|--:|--:|
| p0 | 32,768 | 0.50 | 是／是 | 是 | **否** | 0.156 | 4.0×10⁻³ |
| p1 | 49,152 | 0.30 | 是／是 | 是 | **否** | 0.087 | 3.2×10⁻⁴ |
| p2 | 65,536 | 0.25 | 是／是 | 是 | **否** | 0.076 | 5.1×10⁻⁴ |
| p3 | 65,536 | 0.75 | 是／是 | 否（第 6 個 token 起不同） | **否**（比前 6 個） | 0.0023 | 1.7×10⁻³ |
| p4（只報告） | 122,880 | 0.50 | 是／是 | 是 | **否** | 0.096 | 8.1×10⁻⁴ |

- **A2 對 A1**：5/5 個 prompt、32 個 token 的 token id 與 logprob **全部位元相同**。所以這個設定是確定性的，F1 的差不是雜訊〔實測〕。
- **F1 對 A1**：5/5 個 prompt，32 個 logprob 每一個都不同〔實測〕。
- **F2 對 F1**：5/5 個 prompt、32 個 token **全部位元相同**。`fp8_e4m3` 就是 `fp8` 的別名〔實測〕。兩者都對到 1 byte 的儲存（`vllm/utils/torch_utils.py:39-40`），也對到同一個 fp8 型別（`vllm/v1/attention/ops/chunked_prefill_paged_decode.py:356-357`）〔程式碼〕。
- 針全部找到（5/5，四組都是）。

### 4.6 vLLM 原始碼：ROCm 上的 fp8 KV 有沒有被照做

**backend 怎麼選**（vLLM 0.28.0，`src/vllm-v0.28.0`）
- `vllm/platforms/rocm.py:459-495`：非 MLA 模型的優先順序是 ROCM_ATTN → （AITER 有開才加）→ TRITON_ATTN → TURBOQUANT〔程式碼〕。
- `vllm/platforms/rocm.py:570-616`：每個 backend 都用 `validate_configuration` 檢查，檢查項目包含 kv_cache_dtype。不支援的會被排除，**不會靜默降級**〔程式碼〕。
- `vllm/platforms/rocm.py:686-711`：選有效的 backend 裡優先度最高的一個，並印出九月 log 裡那一行「Found incompatible backend(s) [TURBOQUANT] … Overriding with ROCM_ATTN」（log 標的位置是 rocm.py:703）〔程式碼〕。
- 如果沒有任何有效的 backend，會直接丟 ValueError（`rocm.py:684-688`），不會退回 BF16〔程式碼〕。

**ROCM_ATTN 怎麼處理 fp8**
- 宣告支援：`vllm/v1/attention/backends/rocm_attn.py:171-178`，`supported_kv_cache_dtypes` 含 "fp8"、"fp8_e4m3"、"fp8_e5m2"。所以 fp8 時 ROCM_ATTN 仍然是第一順位，和 BF16 用同一個 backend〔程式碼〕。
- 寫入：`rocm_attn.py:493-501` 呼叫 `PagedAttention.write_to_paged_cache(..., self.kv_cache_dtype, k_scale, v_scale)`，把 K/V 量化成 fp8 存進去〔程式碼〕。
- 讀取：`rocm_attn.py:428-430` 把 cache 當成 fp8 型別讀；MI300X（gfx94x）上是 `float8_e4m3fnuz`（`vllm/platforms/rocm.py:976-985`）。`vllm/v1/attention/ops/chunked_prefill_paged_decode.py:216-217` 讀到 fp8 的 K 時，乘上 k_scale 轉回〔程式碼〕。
- 容量：`vllm/utils/torch_utils.py:39-40` 把 "fp8"、"fp8_e4m3" 對到 1 byte 的 uint8 → 每 token 位元組減半 → 容量 2 倍〔程式碼〕。

**判讀**：ROCm 上的 ROCM_ATTN 有照做 fp8 KV，沒有退回 BF16。log、容量、logprob 三方面都一致〔判讀〕。

### 4.7 工作 B：套判準

| 條件（§2.B） | 結果 |
|:--|:--|
| 容量 fp8 ÷ auto 在 1.8–2.2 | **成立**：F1 2.000×（九月 2.000×） |
| logprob 有差（至少一個 prompt 的前 8 個 token 有不同） | **成立**：4/4 個判定 prompt 都不同 |
| 雜訊對照：A1、A2 本身有差嗎 | 沒有（位元相同），所以 fp8 的差大於雜訊 |
| **判定** | **生效** |

**九月 run 的判定：生效。**
- 九月 log 直接顯示：fp8 存 KV、容量 2.000×、和 BF16 用同一個 backend（ROCM_ATTN）〔實測〕。
- 同一份 vLLM build、同一個模型、同樣的 server 參數下，fp8 會改變 logprob〔實測 20261010-074830-f3-fp8〕。
- 九月沒有存 logprob，九月那 20 題的 logprob 本身 **NOT_MEASURED**。我沒有用九月那 20 題 prompt 重跑（見 §6）。

**為什麼 129K 會 20/20 逐字相同**〔判讀〕
- 答案是一串 7 位數字，模型非常確定：第一個 token 的 logprob 約 −0.015 到 −0.05，也就是機率 95–98%。
- FP8 只讓這個分數動了 0.0003–0.004。這不夠讓 greedy 換一個字。
- 65K 那個 run 有 3/20 不同，很可能和這次的 p3 一樣，是答案之後的文字岔開。九月逐題的原始 CSV 已遺失，所以這點**未查證**。

## 5. 對 D6 和 main.tex 的影響

我沒有改 D6、main.tex 或任何既有檔。下面是建議。

### 對 D6

- **判定由「可能」改為「死路」**（照 D6 §6 的判準原文）。D6 唯一的活路是「INT8 反量化 ≥3.5 ms，而且卡在 GPU 線上」。實測是 0.031 ms，差約 110 倍〔算術〕。
- D6 §3、§4.2 寫成「實測反量化」的 **INT4 0.814、FP8 0.087 ms／chunk** 要改標籤。它們是「attention 讀量化 cache 時的 warm TTFT 差」，INT4 還混了換 kernel 的效果。換成這次的獨立 kernel 值（都 <0.1 ms），D6 的結論不變，甚至更乾淨。
- D6 §4.4 裡「CPU 上量化運算的時間」仍然 NOT_MEASURED。我量的是 GPU 上的反量化，不是 CPU 上的量化。
- D6 §6 第 1 步（有損；品質和位置有沒有關）不受影響，仍然要老師同意才能做。

### 對 main.tex

- **圖 b-tiers 的 caption**（main.tex:1489）寫「GPU 內的精度階反量化成本（FP8 0.17、INT4 1.59 µs/token）」。
  - 這兩個數字就是 §4.3 的 warm TTFT 差。FP8 在雜訊內；INT4 含換 kernel 的效果。
  - 獨立反量化的實測值是 FP8 0.051、INT4 0.053、INT8 0.060 µs/token〔實測 20261010-074630-f3-dequant〕〔算術〕。
  - caption 的結論「比跨裝置搬運低一到兩個數量級」方向不變，甚至更強。但「反量化成本」這個名稱要改，不然就換新數字。
  - 另外，caption 標的來源檔 `results/m2_harness_mi300x/cost_constants_mi300x.csv` 不存在（SUMMARY §4 第 2 項，F4 在處理）。
- **MI300X 上 Llama-3.1-8B 的 FP8 品質結果**（撈針 64K／129K 都是 100%）：FP8 確實有生效，這些結果可以當成「FP8 KV」的結果用。SUMMARY §4 第 3 項可以關閉。
- **新發現的注意事項**：九月 MI300X 的品質 run 裡，fp8_ptk、int8、int4 用 TRITON_ATTN，bf16、fp8 用 ROCM_ATTN。
  - 所以那三種「和 BF16 逐字相同幾題」的比例，混了「換 attention kernel」的效果，不能全部算在量化頭上。
  - 正確率不受影響（換 kernel 不會讓答對變答錯，至少這次沒看到）。但如果 main.tex 要引用「逐字相同」的比例，要註明這一點〔判讀〕。
- main.tex:1455 寫平台 B 用 vLLM v0.28.0，和實際的 server 一致。但九月 run 的 context.txt 記的是 0.19.1 的 commit，F4 稽核時要注意不要被誤導。

## 6. 失敗與異常

**失敗（完整錯誤訊息）**

1. 工作 A：我寫的兩個 FP8 Triton kernel 編譯失敗（`fp8_tensor/triton`、`fp8_ptk/triton`）。原因是 `tl.load(..., other=0)` 的整數 0 不能轉成 fp8。完整 traceback 在 `/mlsteam/data/tiara/runs/20261010-074630-f3-dequant/stdout.log` 和 `f3_dequant_raw.json`，最後幾行是：
   ```
   AssertionError: cannot cast int32[constexpr[4096]] to <['4096'], fp8e4b8>
   triton.compiler.errors.CompilationError: at 5:8:
   def _dq_tensorscale(q_ptr, o_ptr, scale, n, BLOCK: tl.constexpr):
       pid = tl.program_id(0)
       off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
       m = off < n
       q = tl.load(q_ptr + off, mask=m, other=0).to(tl.float32)
           ^
   cannot cast int32[constexpr[4096]] to <['4096'], fp8e4b8>
   ```
   ```
   AssertionError: cannot cast int32[constexpr[64], constexpr[128]] to <['64', '128'], fp8e4b8>
   triton.compiler.errors.CompilationError: at 8:8:
   def _dq_rowscale(q_ptr, s_ptr, o_ptr, n_rows, G: tl.constexpr, BR: tl.constexpr):
       ...
       q = tl.load(q_ptr + off, mask=m, other=0).to(tl.float32)
           ^
   cannot cast int32[constexpr[64], constexpr[128]] to <['64', '128'], fp8e4b8>
   ```
   - 處理：沒有修、沒有重跑。FP8（整個 tensor 一個 scale）已經有 torch.compile（它自己會產生 Triton）和 vLLM `convert_fp8` 兩個數字，而且 FP8 不進判定。CSV 裡這兩列標 `FAILED`。
2. 工作 B：guard 判 `CONTAMINATED_DURING_RUN`（exit 3）。
   - 闖入者只有 pid −1，峰值 182,329 MiB（就是我自己一個 server 的大小），5 個樣本，時間 08:00:29–08:08:01。
   - 對照我自己的 server 關閉時間：A1 最後一行 log 08:00:28，F2 08:08:00。時間對得上。
   - 這是已知的假陽性：自己的行程剛結束時，amd-smi 會報 pid −1。每個 server 開之前 GPU 都是空的（191.46/191.98 GiB free）。
   - 工作 B 不用時間欄，而且容量是在啟動時、GPU 確定是空的時候量的。所以沒有重跑。
3. 工作 A 的 guard：CLEAN，沒有闖入者。
4. 沒有產生 gpucore 檔（repo 和兩個 run 目錄都檢查過）。

**和登記的偏離，或要注意的地方**

1. FP8 per-token-head 只有 plain torch 的數字（Triton 失敗；我也沒做 torch.compile 版本）。不進判定。
2. INT4 的 vLLM 格式版本**沒有做 Hadamard 旋轉（RHT）**。vLLM 真正的 INT4 讀取時要把旋轉轉回去（或把 Q 也轉）。這部分的額外成本 NOT_MEASURED。
3. 計時不含 CPU 發 kernel 的時間（用了 `_sleep` 的技巧）。plain torch 有 2–4 個 kernel，實際呼叫每個 kernel 會多幾 µs，仍然遠小於 1 ms〔判讀〕。
4. 「反量化和重算同時跑」的爭用 NOT_MEASURED（§4.2）。
5. 工作 B 用的是我自己產生的 prompt，不是九月那 20 題（m5_quality.py 已從 `code/` 刪除；要用得從 git 的 e2ff39d 撈）。所以「九月那 20 題的 logprob 有沒有不同」NOT_MEASURED。
6. 工作 B 全部用 `--max-model-len 129536`；九月 65K 那個 run 是 66048。容量和這個參數無關：九月兩個 run 的 BF16 容量都是 1,267,504。
7. 這次的容量（1,271,056）和九月（1,267,504）差 0.3%，因為這次啟動時量到的可用 KV 記憶體是 155.16 GiB（九月 154.73）。倍數不受影響。
8. 我的 log 解析只抓「Using fp8 data type」，所以 F2（fp8_e4m3）的 CSV 欄 `fp8_store_msg` 是 False。實際 log 印的是「Using fp8_e4m3 data type to store kv cache」〔實測 F2/server.log〕。這是我的正規表示式寫太窄，不是 vLLM 沒開。
9. 寫判準前我已經讀了 D6，知道 D6 的 INT4 參考值是 0.814 ms。但判準是照 D6 原文的 1 ms 門檻抄的，不是看了量測結果才定的。
10. 我在 scratchpad 用 CPU 跑過一次 prompt 長度檢查（沒有用 GPU，也沒有產生任何數字）。

**產出檔**
- 程式：`code/m9_f3_dequant.py`、`code/m9_f3_fp8_repro.py`
- 結果：`results/m9_followup/f3_dequant.csv`（48 列）、`results/m9_followup/f3_fp8_repro_gens.csv`（20 列）、`results/m9_followup/f3_fp8_repro_compare.csv`（15 列）。每一列都有 `run_id` 和 `ts`。
- 原始資料（不進 git）：`/mlsteam/data/tiara/runs/20261010-074630-f3-dequant/`（每次的 ms：`f3_dequant_raw.json`）、`/mlsteam/data/tiara/runs/20261010-074830-f3-fp8/`（4 個 server.log、`f3b_results.json` 含 32 個 token 的完整 logprob）。
