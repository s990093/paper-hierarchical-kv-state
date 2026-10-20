# K1：KV 結構（低秩／跨層相似／相鄰 token 相似）怎麼跑（第三階段；10/20 寫）

> **狀態**：分析程式 `code/m16_kvstruct.py` 寫好了，`--self-test` 61 項 ALL PASS（合成資料；run `20261020-043757-k1-selftest`）。**真實的 KV dump 還沒有**：`m16_s1_kvdump.py` 要 GPU，排在 `05_s1_kvdump.sh`。所以**所有結果都是 NOT_MEASURED**，`results/m16_stage3_mi300x/k1_kvstruct/` 目前是空的，也不應該有東西。
> 本頁的時間都是〔估計〕，依據是 §6 的合成資料 bench。
> 寫的時候第二階段正在 GPU 上量時間，所以：
> - 只用 numpy 與標準函式庫；
> - 沒有 import torch／vllm／transformers（程式開頭與結尾都會檢查）；
> - 一律 `OMP_NUM_THREADS=4 nice -n 19 taskset -c 0-3`。
>
> 目的：替兩個新方向做第一步的現象量測。一個是「低秩／降維」，另一個是「token 合併／跨層共用」。文獻整理在 `lit/L5a_lowrank.md`、`lit/L5b_merge_crosslayer.md`。

---

## 0. 檔案

| 檔案 | 狀態 | 內容 |
|:--|:--|:--|
| `code/m16_kvstruct.py` | 新 | 子命令有 `--self-test`、`ropecheck`、`analyze`。只讀 dump，不碰 GPU。 |
| `code/m16_s1_kvdump.py` | **改**（只在 meta.json 補欄位，見 §2） | 要 GPU。這次**沒有執行**，只做了 `py_compile` 和 symtable 的未定義名稱檢查。 |
| `results/m16_stage3_mi300x/k1_kvstruct/` | 空 | 真 dump 出來之後才會寫，見 §4。 |
| `/mlsteam/data/tiara/runs/20261020-043757-k1-selftest/` | 自測的 log | `stdout.log` 最後一行是 `61 項；ALL PASS`。 |
| `/mlsteam/data/tiara/runs/20261020-043758-k1-bench/` | 計時用的 bench | 合成資料，形狀和真的一樣；資料跑完就刪了，只留 log。**不是結果。** |

`context.txt` 記了當時兩支程式的 md5。這兩個 run 都沒有叫 rocm-smi，因為是純 CPU 工作，第二階段又在量時間。

---

## 1. dump 的格式：關鍵事實

10/20 讀原始碼確認，行號以當天的 HEAD 為準。

- **排列**：`kv_bf16.u16` 是 `[n_blocks][L'][2][H][512][D]` 的 BF16 位元，uint16、little-endian。
  - `L'` 是 `meta.layers`。128K 的 dump 只存部分層。
  - 第三維的 0 是 K、1 是 V。
  - 每個 block 來自 `m.chunk_view(i).index_select(0, lidx)`（`m16_s1_kvdump.py`）。
  - KVModel 的 kv 緩衝是 `[L, 2, H, max_len, D]`（`m7_model.py:59`）。
- **位置**：block i 的第 t 個 token 位置是 `i·512 + t`。依據是 `m7_model.py:144–149` 的 `prefill_chunked(start=0)`。
- **K 是 RoPE 之後的**：
  - `m7_model.py:105` 先做 `q, k = self.apply_rope(q, k, cos, sin)`；
  - `:106` 才 `self.kv[li, 0, :, start:end].copy_(k[0])`。
  - V 沒有 RoPE。
- **RoPE 的實作**：照 transformers 5.17.0 原始碼（tiara venv），只讀原始碼，沒有 import。
  - **inv_freq**
    - 預設：`modeling_llama.py:108`、`modeling_qwen2.py:86`。
    - llama3：`modeling_rope_utils.py:642–660`。
    - torch 的 `scalar / tensor` 實際上是 `reciprocal() * scalar`（`torch/_tensor.py:1119–1122`），程式照這個方式模擬 float32 的運算順序。
  - **cos／sin**：`modeling_llama.py:122–127`。
    - 角度是 float32 乘法（K=1 的 matmul，只捨入一次）。
    - `cat(freqs, freqs)` 之後轉成 **BF16**。
  - **套用**：`rotate_half`（`:130–134`）把**第 j 維和第 j+D/2 維配成一對**，不是相鄰兩維。
    - `k*cos + rotate_half(k)*sin`（`:159`）每一步都在 BF16 算。
    - KVModel 對 Qwen 也用 Llama 的 `apply_rotary_pos_emb`（`m7_model.py:42`），兩邊的式子相同。
- **兩個模型的 RoPE 參數**（從 config.json 讀，不是猜的）：
  - **Llama-3.1-8B-Instruct**（`models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75…/config.json`）：`rope_theta` 500000；`rope_scaling` = llama3，factor 8、low_freq_factor 1、high_freq_factor 4、original_max_position_embeddings 8192；head_dim 128、8 個 KV head、32 層。
  - **Qwen2.5-7B-Instruct-1M**：`rope_theta` 1e7，`rope_scaling` null，所以是預設 RoPE。4 個 KV head、28 層。
    - KVModel 載入的是原版 snapshot `e28526f7…`（`m10_tasks.py:118`）。noDCA 版的 RoPE 參數和原版相同。
    - HF 的 Qwen2 沒有 DCA 的程式碼（`modeling_qwen2.py` 裡找不到 `dual_chunk`），所以 `dual_chunk_attention_config` 在這條路徑上不起作用。
- **逆旋轉**：每一對是 `[[c, −s], [s, c]]`。用 dump 實際用的那組 BF16 c、s 做精確的反矩陣，也就是除以 c²+s²，以 float64 計算。
  - RoPE 本身是正交旋轉，可以精確逆轉。
  - 但 dump 的值有 BF16 捨入（每個值的相對誤差 ≤ 2^-8），所以逆回來的 K 和真正 RoPE 前的 K 之間有 BF16 捨入量級的差距。自測量到每個向量的相對誤差最大 3.2e-3，**不是逐位元相同**。

---

## 2. 對 `code/m16_s1_kvdump.py` 的改動（只補 meta.json 的欄位）

`kv_bf16.u16` 和各格式的 `.bin` **一個位元組都沒動**。`m16_s1_lossless.py` 只讀固定幾個 key，新加的 key 不影響它。

| 新欄位 | 內容 |
|:--|:--|
| `kv_layout` | 上面那段排列的文字說明 |
| `pos_start` | 0 |
| `k_rope` / `v_rope` | `"post"` / `"none"` |
| `transformers` | 版本字串 |
| `rope` | 從 `m.hf.model.rotary_emb` 讀出實際用的值：`inv_freq`（64 個 float，**是 buffer 本身，不是用公式算的**）、`inv_freq_dtype`、`attention_scaling`、`rope_type`、`rope_parameters`（`m.cfg.rope_parameters`）、`pairing="rotate_half"` |
| `rope_probe` | 第 0 層 K 的探針，見下面 |

`rope_probe` 的內容：
- 取最多 6 個位置：0、1、511、512、n/2、n−1。
- 每個位置存四樣東西，都是 hex 字串，Llama 合計約 50 KB：
  - **RoPE 前**的 K `[H, D]`；
  - **RoPE 後**的 K `[H, D]`；
  - torch 算出的 BF16 cos、sin `[D]`。
- 計算方式和 prefill 用同樣的形狀：整個 512-token chunk 重算第 0 層的 embedding → RMSNorm → k_proj，再用同一個 `rotary_emb` 和 `apply_rope`。
  - 第 0 層的 K 只看 token 本身、和上下文無關，所以 RoPE 後的探針應該和 dump **逐位元相同**。
  - 額外的 GPU 工作只有至多 4 個 chunk 的單層投影，可以忽略。
- 兩段都包在 `try/except` 裡：
  - 失敗時把 `{"error": "..."}` 寫進 meta，**不會讓 dump 失敗**；
  - stdout 最後會多印一行 `rope_probe: ok | rope: ok`，失敗時印錯誤字串；
  - 分析端看到 error，會改用資料自身的核對（§3.1），並把原因寫進 `rope_check.json`。

**沒有在 GPU 上跑過。** `00_smoke.sh` 會用 4 個 block 跑一次 kvdump，那是第一次真正執行。冒煙之後先做 §4 的步驟 1。

---

## 3. 量什麼（定義寫死）

### 3.1 RoPE 核對（先做；沒過就停，exit 2，規則 2）

規則 6 的精神是「用資料自身驗算，不看欄位名推斷」。

**1. 探針**（新 dump 才有）。只要有一項不過，就是 FAIL：
- (a) dump 第 0 層的 K 在探針位置 ＝ 探針的 RoPE 後 K。容許 ≤ 2 ulp，而且 ≥ 99% 逐位元相同。這一項核對排列、位置、以及「K 是 RoPE 後」。
- (b) 程式模擬的 BF16 cos／sin ＝ torch 算的（≤ 1 ulp）。這一項核對 inv_freq、float32 乘法和 `cat(freqs, freqs)`。
- (c) 程式模擬的 BF16 前向旋轉作用在探針的 RoPE 前 K 上，結果 ＝ 探針的 RoPE 後 K（≤ 1 ulp）。這一項核對配對方式和正負號。
- (d) 程式的逆旋轉作用在 dump 上，結果 ≈ 探針的 RoPE 前 K（每個向量的相對誤差 ≤ 0.02）。這一項核對分析實際走的那條路徑。

**2. 資料自身**（每個 dump 都做）。
- 想法：RoPE 前的 K 如果有不隨位置轉動的平均方向，RoPE 後這個平均會被轉散。
- 指標：M ＝ ‖平均 token‖² ／ 平均 ‖token‖²，在逆旋轉之後計算。
- 對照組：不逆轉、轉反方向、相鄰兩維配對、θ×2、θ×½、llama3 不縮放、inv_freq 捨成 BF16。
- 判定：
  - **PASS**：期望的 M 不低於每個對照組 −0.001，而且比「不逆轉」高至少 0.005。
  - **INCONCLUSIVE**：比「不逆轉」高不到 0.005。
- **這個方法依賴一個假設**：Qwen2 的 k_proj 有 bias（`modeling_qwen2.py:190`），這點確定；Llama 沒有 bias，靠的是文獻說 K 有固定的大值通道。資料裡沒有夠大的平均方向時，結果會是 INCONCLUSIVE。

**3. 判定方式**：
- 探針全過 → `PASS_PROBE`。這時資料核對只當旁證，不過的話寫進 warnings。
- 沒有探針（或 dump 沒有第 0 層）→ 資料核對必須是 PASS，否則 FAIL。
- 想跳過 RoPE 前的量測、只做其餘部分，要加 `--no-pre-rope`。**要不要這樣做由人決定。**

### 3.2 低秩（每層、每個 KV head；K_post、K_pre、V 分開）

**計算方式**：Gram 矩陣 XᵀX 串流累加，記憶體和 token 數無關。最後做特徵分解，特徵值就是 σ²。

**每個 rank r 都報這些**：
- 能量比 Σ_{i≤r}σ²／Σσ²。
- 相對誤差 ‖X−X_r‖_F／‖X‖_F ＝ √(1−能量)，依據 Eckart–Young。
- centered 版：先減掉 token 平均再降秩，誤差仍然除以 ‖X‖_F。
- 預設 r 是 8、16、32、64、96。

**每個（層, head）另外報**：
- r90／r95／r99（含 centered 版）、stable rank；
- 平均方向佔的能量比；
- 單一 token 佔的最大能量比和它的位置，用來看 attention sink 會不會主導；
- 必要時加 `--skip-first-tokens k` 重跑一次比較。

**校準 → 測試**：
- 用前 `--calib-tokens`（預設 4096）個 token 算出基底，套到其後的 token。
  - uncentered：取前 r 個特徵向量。
  - centered：校準集的平均，加上校準共變異的前 r 個特徵向量。
- 同時報測試集自己的最佳值（oracle）。xfer 和 oracle 的差，就是基底的泛化損失。
- 校準集和測試集**分開**，所以不會高估。

**整層串接（`head=all`）**：
- 把 H 個 head 串成 H·D 維一起降秩，rank ＝ r·H，和逐 head 的參數量相同。
- 這是這份資料上「同參數量下任何線性投影（含分組 head）」的誤差下限，**不是**某個方法的實測。

**同位元組的換算**〔算術，不是量測〕：
- rank r 的 BF16 係數是 r/128 的位元組：r=16 是 1/8，r=32 是 1/4，r=64 是 1/2。
- 基底是 r×128×2 B／（層, head）。T=32K 時約多 0.4%，可以忽略。
- 對照量化格式的實際位元組比（`c1a_s2_howto.md` §2）：INT4 KIVI 是 0.3125，K4V2 是 0.25。

### 3.3 跨層相似（MiniCache 的量法）

- 同一個 token、同一個 head，層 l 和 l+1 先各自做方向正規化（除以長度），再求 cosine。另外報整層 H 個 head 串接後的 cosine（`head=all`）。
- **K 的跨層 cosine 在 RoPE 前後相同**：同一個位置套的是同一個正交旋轉，自測有驗證。所以只算一次，標成 K。
  - 另外用 RoPE 前的 K 再算一次當旁證，最大差寫在 `summary.json` 的 `xlayer_K_pre_vs_post_max_abs_diff`。
  - 理論上兩者只差 BF16 捨入。差距大，代表逆旋轉有問題。
- **只有 l 和 l+1 都在 dump 裡才算。**
  - `05_s1_kvdump.sh` 的 128K dump 只存了 0,8,16,24,31（Llama）和 0,7,14,21,27（Qwen），**沒有相鄰層**，所以 128K 的跨層相似是 NOT_MEASURED。
  - 32K 的 dump 是全層，可以量。
  - 如果要 128K 的跨層數字，得另外 dump 成對的層，例如 `--layers 0,1,8,9,16,17,24,25,30,31`。這是建議，job 檔我沒改。

### 3.4 相鄰 token 相似

- 同一層、同一 head，token t 和 t+1 的 cosine，K_post、K_pre、V 分開算。
- K_post 是快取裡實際存的、合併演算法實際會合的東西；K_pre 是去掉位置旋轉之後的。
- block 邊界的那一對，只在兩個 block 都讀到時才算。

### 3.5 中心化（第二輪；`--no-centered` 可關掉）

- 先減掉第一輪算出的每（層, head）token 平均。K 用 RoPE 前的平均；RoPE 後減全域平均沒有意義。
- 然後重算 3.3 和 3.4，結果標成 `K_centered`／`V_centered`／`K_pre_centered`。
- 為什麼要做：K 有固定的大通道（Qwen 還有 bias），未中心化的 cosine 可能只是「大家都指向平均方向」。L5b 的 P1 也建議這樣做。
- 自測裡有一組合成資料：未中心化的平均 cosine 是 0.56，中心化後是 0.02。

### 3.6 cosine 分布報哪些數字

| 項目 | 怎麼算 |
|:--|:--|
| 平均、標準差 | 直接累加 |
| 角距離平均 | arccos/π，即 MiniCache 的 d |
| 分位數 p05、p10、p25、p50、p75、p90、p95 | 0.001 寬的直方圖內插，誤差 ≤ 0.001 |
| ≥ 門檻的比例 | **逐值精確比較**，不經過直方圖。預設門檻：0.75（KVMerger 的門檻）、0.8、0.9、0.95、0.99 |
| `n_zero` | 長度為 0 的向量不算 cosine，另外記數 |

### 3.7 輸出（`--out`）

- **CSV**：
  - `lowrank.csv`：每個 dump × 層 × head × 種類 × rank 一列；
  - `lowrank_eff.csv`：有效秩等；
  - `xlayer.csv`、`adjtok.csv`。
- **JSON**：`lowrank.json`、`xlayer.json`、`adjtok.json`（彙總），以及 `rope_check.json`、`summary.json`（設定、時間、峰值 RSS）。
- **給人看的**：`summary.md`。內容有 RoPE 核對表、每種 K／V 的低秩表、跨層表（含最像的 3 對）、相鄰 token 表。
- **每個 CSV 都有 `run_id`、`ts`**，`run_id` 來自環境變數 RUN_ID，m7run 會自動帶。
- `--spectra` 可以另存完整特徵值（npz，每個 dump 約 1–2 MB）。這個檔放 `/mlsteam/data/tiara/...`，不進 git。

---

## 4. 真 dump 出來後要跑的指令

```bash
cd /mlsteam/workspace/paper-hierarchical-kv-state
source code/m7_env.sh                       # tiara venv（只用到 numpy）、TIARA_RUNS、m7run
KVD=/var/tmp/tiara_s3/kv_dumps              # 05_s1_kvdump.sh 的輸出（本地盤）
OUT=results/m16_stage3_mi300x/k1_kvstruct
CPU="env OMP_NUM_THREADS=4 nice -n 19 taskset -c 0-3 python -I"

# 0. 自測（幾秒鐘）
$CPU code/m16_kvstruct.py --self-test                                   # 要看到「61 項；ALL PASS」

# 1. 冒煙：00_smoke.sh 跑完 kvdump（4 個 block）之後
grep "rope_probe" $TIARA_RUNS/$(/bin/ls -t $TIARA_RUNS | grep -m1 m16-smoke-kvdump)/stdout.log   # 要是 rope_probe: ok | rope: ok
$CPU code/m16_kvstruct.py ropecheck --dump /mlsteam/data/tiara/stage3/smoke/kv_llama_t4_4         # 要 PASS_PROBE、exit 0
#    不是 PASS_PROBE → 停，把 ropecheck 印出的 JSON 回報（規則 2）。不要直接加 --no-pre-rope 繞過。

# 2. 正式：05_s1_kvdump.sh 的六份 dump
mkdir -p /mlsteam/data/tiara/stage3/k1
m7run m16-k1-kvstruct $CPU code/m16_kvstruct.py analyze \
  --dump $KVD/llama_T4_64 $KVD/llama_T2_64 $KVD/qwen_T4_64 $KVD/qwen_T2_64 $KVD/llama_T4_252 $KVD/qwen_T4_252 \
  --out $OUT --spectra /mlsteam/data/tiara/stage3/k1/spectra_$(date -u +%Y%m%dT%H%MZ).npz
#    exit 0＝寫好；exit 2＝某個 dump 的 RoPE 核對沒過（看 $OUT/rope_check.json，停、回報）；exit 3＝記憶體估計超過 --max-mem-gb

# 3.（選做）attention sink 有沒有主導低秩能量：去掉位置 0 再跑一次，只看低秩
m7run m16-k1-kvstruct-skip1 $CPU code/m16_kvstruct.py analyze \
  --dump $KVD/llama_T4_64 $KVD/qwen_T4_64 --out $OUT/skip1 --skip-first-tokens 1 --no-centered
```

補充說明：
- `m7run` 走 `runsh`，它寫 `context.txt` 時會叫 `rocm-smi`。這和 `35_s1_cpubench.sh` 這類 CPU job 的做法一樣。
  - 如果要在第二階段**量時間的 stage** 執行，又不想碰任何 GPU 工具，就照 CLAUDE.md §4.1 自己建 run 目錄，不叫 rocm-smi。§0 的兩個 run 就是這樣做的。
- 也可以只看部分資料：
  - `--layers`、`--heads` 選層和 head；
  - `--max-blocks N` 在校準之後平均抽 N 個 block；
  - `--no-joint` 不算整層串接，最省記憶體和時間。
- T2 是 RULER 的重複句子，相鄰 token 和低秩的數字可能特別好看，**不能代表一般文字**。每個 dump 分開報，T4 才是主要的數字。

---

## 5. 讀結果時要對照的東西

go／no-go 的門檻由人決定，這裡不另訂。

- **低秩**：
  - 主要看 **K_pre 的 xfer 誤差**，不是 oracle。前者才是真的「先校準、後使用」。也要看它和 K_post 的差距；L5a 整理的文獻說 pre-RoPE 最低秩（〔文獻〕，我們還沒量）。
  - 同位元組要拿 §3.2 的換算去對量化格式。
  - L5a 的 E-L1 已經有停止條件的提案：「低秩誤差在 ≥80% 的（層, head）都大於同位元組量化誤差 → 不做 E-L2」。
  - 但這支**沒有**算量化格式的同一種誤差（‖X−Q(X)‖_F／‖X‖_F）。要比的話，得在同一份 dump 上另外補。
- **跨層**：
  - 要同時看未中心化和中心化。如果只有未中心化的高，代表「像」來自平均方向，對合併沒什麼用。
  - L5b 整理的第三方結果（xKV）說，Llama-3.1-8B 相鄰層逐 token 的 cosine「出奇地低」〔文獻，不是我們量的〕。這次的數字可以直接對照。
- **相鄰 token**：
  - 主要看 K_post 的 ≥0.75（KVMerger 的門檻）和 ≥0.9、≥0.95 的比例，以及 T4 和 T2 的差。
- **這些都是 KV 本身的幾何量，不是品質。**
  - K 的誤差要經過 q·k 和 softmax 才會變成 attention 的誤差。
  - cosine 高也不等於合併不掉分；MiniCache 還會保留長度和最不像的 token。
  - 品質要另外做實驗。

這一版**沒有做**的：L5a 的 E-L1 和 L5b §8 提的擴充，它們都可以加在這支程式上，不需要另寫一支。
- 跨 prompt 的基底（T4 的基底套到 T2）；
- 每個 block 各自做 SVD；
- 對 KIVI 的同位元組誤差；
- 跨層 CKA；
- 多層串接的聯合秩。

---

## 6. 時間與記憶體〔估計〕

**bench 的條件**：
- run `20261020-043758-k1-bench`，指令在 `cmd.sh`。
- 合成 dump 的形狀和真的一樣：
  - Llama：4 個 block × 32 層 × 8 head × 128；
  - Qwen：4 個 block × 28 層 × 4 head × 128。
- 在第 0–3 核執行，nice 19、4 個 thread，當時 load average 約 17（第二階段正在跑）。

**bench 量到的**：

| 形狀 | 第一輪 | 第二輪（中心化） | 結尾（特徵分解等） | 總時間 | 峰值 RSS | ropecheck |
|:--|--:|--:|--:|--:|--:|--:|
| Llama | 5.9 s（1.5 s／block） | 1.0 s | 約 48 s（32 層 × 3 種 × 整層 1024 維） | 55.3 s | 2.19 GiB | 2.3 s |
| Qwen | 1.9 s | 0.5 s | 約 10 s | 12.1 s | 0.67 GiB | 1.2 s |

**照比例換算到六份 dump**〔估計〕：

| dump | block × 層 | 估計時間 |
|:--|:--|--:|
| llama_T4_64、llama_T2_64 | 64 × 32 | 各約 2.8 分 |
| qwen_T4_64、qwen_T2_64 | 64 × 28 | 各約 0.9 分 |
| llama_T4_252 | 252 × 5 | 約 1.3 分 |
| qwen_T4_252 | 252 × 5 | 約 0.5 分 |
| **合計** | | **約 10 分鐘（4 核）** |

- 兩輪各讀一次 dump，合計約 30 GiB，從本地盤讀，另外加幾十秒。
- 機器忙的時候保守估 **20 分鐘**。
- 記憶體峰值約 2.2 GiB（Llama 全層加整層串接），程式自己的估計是 3.15 GiB，遠低於 16 GB。

**曾經抓到並修好的效能問題**：`np.linalg.eigh` 回傳的特徵向量反轉之後是負 stride，matmul 不會走 BLAS，1024 維一組要 12.5 s。改成連續記憶體，只需要特徵值的地方也改用 `eigvalsh` 之後，降到 0.75 s。

---

## 7. 自測涵蓋什麼（61 項；`python -I code/m16_kvstruct.py --self-test`）

| 類別 | 驗什麼 |
|:--|:--|
| BF16 | round-to-nearest-even（含 tie 到偶數）、捨入誤差 ≤ 2^-8、位元往返、ulp 距離 |
| inv_freq | float32 模擬和獨立的 float64 參考實作（Meta 的 if／elif 寫法）相差 ≤ 1e-6；llama3 的三段（不變／÷8／中間）；config 的 v4 和 v5 兩種寫法解析結果一致 |
| RoPE | float64 旋轉後再逆旋轉 ≤ 1e-12；正交（保長度）；配對方式（e_0 只轉到第 0 維和第 D/2 維）；相對位置性質 ⟨R_m q, R_n k⟩ 只看 m−n；BF16 前向模擬後再逆轉，每個向量的相對誤差 ≤ 2^-7（實測最大 3.2e-3） |
| 低秩 | 已知奇異值的誤差公式；秩 5 的矩陣；Gram 特徵值 ＝ SVD 的 σ²；分 block 串流 ＝ 整段 SVD；校準 → 測試在同子空間時 ≈0、在正交子空間時 ＝1；uncentered 和 centered 都和直接投影一致；「平均＋秩 2」的 centered 和 uncentered 差異 |
| cosine | 已知 cos α 的序列（含 block 邊界）；≥ 門檻逐值精確；已知混合分布的分位數；長度為 0 的向量；角距離 ＝ α/π；跨層 cosine 在 RoPE 前後相同（精確 cos/sin 時 ≤ 1e-12，BF16 cos/sin 時 ≤ 1e-2） |
| 假 dump：探針與門檻 | 探針 (a)–(d) 全過、資料核對 PASS → PASS_PROBE；舊 dump（沒探針、沒 meta.rope）用 config 加資料核對 → PASS_DATA；dump 其實沒轉時，探針抓到 → FAIL；沒探針而 dump 沒轉時，資料核對抓到 → FAIL；沒探針而 θ 錯（dump 用 1e6，config 寫 5e5）→ FAIL |
| 端到端 | CSV 都有 run_id、ts；已知結構量得出來：K_pre 秩 4、K_post 秩變高、V 秩 4、跨層 1→2 很像而 0→1 不像、漂移序列的相鄰 token 很像、中心化後 K_pre 的相鄰相似消失；缺 l+1 層的配對寫 NOT_MEASURED；RoPE 核對不過時 exit 2 且不寫 lowrank.csv；`--no-pre-rope` 照跑其餘部分；層、head、block 抽樣；`--skip-first-tokens`；spectra 檔；記憶體估計；全程沒有 import torch／vllm／transformers |
