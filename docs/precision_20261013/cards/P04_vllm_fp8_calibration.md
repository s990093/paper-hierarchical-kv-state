# P04｜vLLM 0.28.0 的 FP8 KV：scale 從哪來、3090 走哪條路、怎樣才算「正確校準」

> **2026-10-13 獨立複核更正**（見 [V_verification_20261013.md](V_verification_20261013.md)）：(1) 移除前的 `--calculate-kv-scales`：Q、K 是 max÷200，**V 是 max÷100**（`envs.py:1328-1333`）。(2) v0.28.0 的 scale 來源不只「checkpoint 或 1.0」：`*_per_token_head` 會在寫入時動態算，但只有 TRITON_ATTN 支援，而它在 SM89 以下拒收所有 fp8*（`triton_attn.py:540-549`），所以 3090 上只有 INT8 動態版能用。(3) fork #28 的 0.031 要「減平均＋每 (head, channel) scale」兩步一起做；只減平均是 0.070。下文保留原句，以這段為準。

> 一句話：vLLM 0.28.0 的 FP8 KV 只有兩種 scale 來源：checkpoint 裡的 per-tensor scale，或預設 1.0。`--calculate-kv-scales` 已經在 0.28.0 之前移除。3090（sm_86）上唯一能跑原生 FP8 KV 的是 FlashInfer，而且只能用 per-tensor scale，讀出來會先轉成 BF16／FP32 再算。超出 ±448 的值會被截在 448，不會變 NaN。**對 Qwen2.5 來說，「沒校準所以溢位」這個假設證據很弱**：外部量測的 max|K| 約 423（< 448），而我們自己在 MI300X 上「不可能溢位」的逐 token 逐 head 動態 FP8 也只拿 0/20。比較像是 K 有一個很大的固定偏移，FP8 只有 3 位尾數，大值附近的捨入誤差蓋過了有用的訊號。這種問題 per-tensor 校準修不好。

- **日期**：2026-10-13。
- **版本**：vLLM tag `v0.28.0` 存在，commit `2cf0a69`，2026-08-24〔程式碼 `gh api repos/vllm-project/vllm/git/refs/tags/v0.28.0`〕。原始碼是在 scratchpad 淺 clone 這個 tag 後讀的。這版 vLLM 釘死 `torch==2.13.0`、`flashinfer-python==0.6.16.post3`〔程式碼 vllm@v0.28.0 requirements/cuda.txt:7、:17〕。
- **其他讀過的程式碼**：FlashInfer `v0.6.16.post3`（`flashinfer/prefill.py`、`include/flashinfer/attention/prefill.cuh`、`decode.cuh`）；llm-compressor `0.14.0`（clone）；compressed-tensors `main`（2026-10-13，只讀一個檔）；PyTorch `v2.13.0` 與 `v2.9.0` 的 `Float8_e4m3fn.h`。
- **標記**：〔原文 §／p.〕＝論文；〔程式碼 repo@tag file:line〕；〔文件 URL〕＝官方文件、GitHub issue／PR、HF 頁面；〔實測 run_id〕＝本專案已有的量測；〔判讀〕＝我的推論；〔未查證〕＝沒讀到或沒跑過。
- **複核**：本卡是 self-check（逐行回到原始碼或頁面核對），**不是**獨立 reviewer 的複核。

---

## 1. k_scale／v_scale 怎麼設

### 1.1 checkpoint 沒有 scale 時：一律 1.0

- `--kv-cache-dtype fp8` 在 CUDA 上就是 `fp8_e4m3`〔程式碼 vllm@v0.28.0 vllm/config/cache.py:77-86〕。
- Attention 層建立時，先註冊 `_k_scale`、`_v_scale`、`_q_scale`、`_prob_scale` 四個 buffer，值都是 1.0〔程式碼 vllm@v0.28.0 vllm/model_executor/layers/attention/attention.py:124-145、:179〕。
- 只有當 quant config 對 Attention 層回傳「會載入權重」的 quant method，才會建立可載入的 scale 參數〔程式碼 同檔 :88-92、:185-215〕。
- 沒有 quant config（純 BF16 checkpoint），載完權重後再設回 1.0〔程式碼 同檔 :580-592〕。
- 官方文件也寫：不校準時「All quantization scales are set to 1.0」〔程式碼 vllm@v0.28.0 docs/features/quantization/quantized_kvcache.md:25-30〕。
- 啟動時只會印一行 info：「may cause accuracy drop without a proper scaling factor」〔程式碼 vllm@v0.28.0 vllm/config/cache.py:281-288〕。
- **AWQ 權重也是 1.0**：AutoAWQ 的 `get_quant_method` 只處理 Linear 與 LM head〔程式碼 vllm@v0.28.0 vllm/model_executor/layers/quantization/auto_awq.py:285-330〕。整個 quantization 目錄裡，有 KV scale 方法（`BaseKVCacheMethod`）的只有 fp8、modelopt、quark、compressed-tensors 四種〔程式碼 grep `BaseKVCacheMethod` vllm/model_executor/layers/quantization/〕。
  - 所以 8/31 那次 3090 的 Qwen AWQ 撈針（FP8 1/20），scale 應該是 1.0〔判讀：從程式碼推；那次 run 的 server log 我沒有重看，vLLM 版本也沒重查〕。

### 1.2 `--calculate-kv-scales`：0.28.0 已經沒有了

- 0.28.0 的 `CacheDType` 與 `CacheConfig` 裡都沒有這個選項〔程式碼 vllm@v0.28.0 vllm/config/cache.py:19-37、:44-198〕。全 repo grep `calculate_kv_scales`，只剩一行註解〔程式碼 vllm@v0.28.0 vllm/models/kimi_k3/nvidia/mla.py:445〕。
- 移除它的是 PR #49389，2026-08-03 合併，merge commit `dd11df0`〔文件 https://github.com/vllm-project/vllm/pull/49389〕。v0.28.0 比這個 commit 新 580 個 commit〔程式碼 `gh api repos/vllm-project/vllm/compare/dd11df0...v0.28.0`：ahead_by 580、behind_by 0〕。
- PR 說明：移除後 scale 只剩一條路，「checkpoint 有就載入，沒有就 1.0」〔文件 同上 PR〕。
- 移除前（parent `0cf49a5`）它做的事：
  - 早就標成 deprecated，說會在 v0.19 移除〔程式碼 vllm@0cf49a5 vllm/config/cache.py:111-115、:273-283〕。
  - **只算一次、整個 tensor 一個值**：第一次 forward 時，`scale = max|x| / 常數`，算完就把旗標關掉〔程式碼 vllm@0cf49a5 vllm/model_executor/layers/attention/attention.py:584-594〕。
  - 常數預設 Q 200、K 200、V 100〔程式碼 vllm@0cf49a5 vllm/envs.py:1328-1333〕。也就是說，最大值會被放在 200，不是 448。
  - 用的是「第一個 batch」的資料，不是校準資料集〔判讀：從 :584-594 只看當次 forward 的 query/key/value 推得；第一個 batch 是不是 profiling 的假資料，我沒查〕。

### 1.3 從 checkpoint 載入（llm-compressor／compressed-tensors）

- config 裡有 `kv_cache_scheme`，而使用者沒指定 dtype（`auto`）時，自動改成 `fp8`；使用者明確指定（例如 `bfloat16`）時以使用者為準〔程式碼 vllm@v0.28.0 attention.py:271-281〕。
- compressed-tensors：Attention 層一律拿到 `CompressedTensorsKVCacheMethod`〔程式碼 vllm@v0.28.0 vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:207-208〕。
  - 只接受 `type: float`、`num_bits: 8`、`strategy` 是 `tensor` 或 `attn_head`、`symmetric: true`〔程式碼 同檔 :1053-1081〕。
  - 只針對 Attention 的 config group 會被跳過，所以「只量化 KV」的 checkpoint，權重保持 BF16〔程式碼 同檔 :229-247〕。
  - 權重名稱 `model.layers.N.self_attn.k_scale` 會被改成 `...self_attn.attn.k_scale`〔程式碼 vllm@v0.28.0 vllm/model_executor/model_loader/weight_utils.py:1436-1437〕。
  - `attn_head` 策略：每個 KV head 一個 scale；但給 attention backend 用的 `_k_scale_float` 是**所有 head 取最大**〔程式碼 compressed_tensors.py:1097、:1201-1209〕。
  - checkpoint 沒有 `q_scale` 時，`q_scale` 初始化為 1〔程式碼 同檔 :1105-1107〕。
- 其他格式（`quant_method: fp8`、quark、modelopt）走 `BaseKVCacheMethod`：
  - 先設成 −1（無效值），載到才覆蓋〔程式碼 vllm@v0.28.0 vllm/model_executor/layers/quantization/kv_cache.py:18-39、:57-69〕。
  - 兩個都沒載到 → 1.0，並警告「Using KV cache scaling factor 1.0 for fp8_e4m3」〔程式碼 同檔 :107-111、:146-151〕。
  - 只支援 per-tensor〔程式碼 同檔 :124-127〕。

### 1.4 有沒有 per-head、per-token 的 FP8？

- 有三種「每 token 每 head 動態 scale」的 dtype：`fp8_per_token_head`、`int8_per_token_head`、`int4_per_token_head`〔程式碼 vllm@v0.28.0 vllm/config/cache.py:19-37；vllm/v1/kv_cache_interface.py:43-53、:80-98〕。
  - 寫入時每個 (token, head) 算 `scale = absmax / QUANT_MAX`，再 clamp〔程式碼 vllm@v0.28.0 vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:143-153、:207-220〕。
  - 靜態 scale 這時被固定成 1.0〔程式碼 kv_cache.py:82-93〕。
  - **只有 TRITON_ATTN 支援這三種**〔程式碼 vllm@v0.28.0 vllm/v1/attention/backends/triton_attn.py:296-306〕。FlashInfer、FlashAttention 的清單裡都沒有〔程式碼 flashinfer.py:408-417；flash_attn.py:79-85〕。
- 靜態 per-head（`attn_head`）需要 FlashAttention 3 以上〔程式碼 flash_attn.py:125-128〕。其他 backend 預設不支援，選擇時會被排除〔程式碼 vllm@v0.28.0 vllm/v1/attention/backend.py:271-272、:393-394〕。
- 另有逐層跳過量化：`--kv-cache-dtype-skip-layers`〔程式碼 vllm/config/cache.py:114-116；attention.py:289-311〕。這是**逐層**混精度。
- **沒有逐 block 或依位置的混精度選項**〔判讀：`CacheDType` 與 attention.py 裡沒有任何依 token 位置或 block 決定 dtype 的路徑〕。

---

## 2. 3090（sm_86）上哪個 backend 收 FP8 KV

| backend | 收 FP8 KV？ | 在 sm_86 上 | 依據 |
|:--|:--|:--|:--|
| FLASH_ATTN | 只在 FA3＋SM90 或 FA4＋SM100 | **不收**。sm_86 退回 FA2，理由字串「FP8 KV cache requires FA3 on SM90 or FA4 on SM100」 | 〔程式碼 vllm@v0.28.0 vllm/v1/attention/backends/fa_utils.py:91-100、:223-243；flash_attn.py:222-234〕 |
| FLASHINFER | `fp8`、`fp8_e4m3`、`fp8_e5m2` | **收**，能力範圍 8.0–12.1 | 〔程式碼 flashinfer.py:408-417、:531-540〕 |
| TRITON_ATTN | `fp8`／`e4m3`／`e5m2`＋三種 per-token-head | 選擇時會被當成有效（能力檢查回傳 True），但 per-tensor FP8 的寫入 kernel 會 assert「needs native fp8e4nv (SM89+)」 | 〔程式碼 triton_attn.py:431-432；triton_reshape_and_cache_flash.py:20-30、:397-401〕 |

- **預設順序**（非 Blackwell）：FLASH_ATTN → FLASHINFER → TRITON_ATTN → FLEX_ATTENTION → TURBOQUANT，取第一個有效的〔程式碼 vllm@v0.28.0 vllm/platforms/cuda.py:156-163、:431-463〕。
- 所以 3090 上 `--kv-cache-dtype fp8` 預設選 **FLASHINFER**〔判讀：FA 被排除後的下一個；前提是 FlashInfer 裝得起來。若 FlashInfer 不能用，vLLM 不會自動退開——issue #60262 在 sm_120 上就是選了不能用的 FlashInfer 然後當掉〔文件 https://github.com/vllm-project/vllm/issues/60262〕〕。
- **BF16 預設選 FLASH_ATTN（FA2），FP8 選 FLASHINFER：兩者不是同一個 kernel**〔判讀：由上表推〕。要「同樣的 attention 路徑」，BF16 也要加 `--attention-backend FLASHINFER`〔程式碼 vllm@v0.28.0 vllm/engine/arg_utils.py:965〕。
- `fp8_per_token_head` 在 sm_86 能不能跑：**未查證**。它也用 Triton 存進 fp8 指標；vLLM 自己在 per-tensor 路徑寫了「需要 SM89+」，我推測這條也會失敗〔判讀〕。`int8_per_token_head` 在 3090 跑過（8/31，19/20）〔實測 20260831-181930-m5-needle，見 research_20261010_directions/D6_precision.md:239〕。
- `fp8_per_token_head`＋OffloadingConnector 曾經一開始就產生亂碼（#48412），修正 PR #49226 在 2026-07-26 合併〔文件 https://github.com/vllm-project/vllm/issues/48412〕，v0.28.0 已包含（比 merge commit `48ebd6f` 新 935 個 commit）〔程式碼 `gh api compare`〕。

### 2.1 FlashInfer 在 sm_86 上怎麼算

- **Q 保持 BF16**：只有 SM90 或 SM100 家族才把 Q 轉成 FP8；註解寫「Architectures with only fa2 … cannot consume FP8 queries」〔程式碼 vllm@v0.28.0 flashinfer.py:952-961〕。query 量化也只在 SM100 家族開〔程式碼 同檔 :1799-1805〕。
- **prefill**：FP8 的 K/V tile 先在 shared memory 轉成 Q 的 dtype（BF16），再用 BF16 tensor core 算〔程式碼 flashinfer@v0.6.16.post3 include/flashinfer/attention/prefill.cuh:1098-1124、:1172、:1252〕。
- **decode**（非 tensor-core 版）：K/V 用 `cast_load` 讀成 FP32 向量再算〔程式碼 flashinfer@v0.6.16.post3 include/flashinfer/attention/decode.cuh:72-79、:137-138〕。
- scale 怎麼用：`k_scale`（和 `q_scale`）乘進 softmax 的 `sm_scale`；`v_scale` 乘在輸出上〔程式碼 flashinfer@v0.6.16.post3 flashinfer/prefill.py:2719-2723、:2891-2897〕。vLLM 每次都把這三個值傳進去〔程式碼 vllm@v0.28.0 flashinfer.py:2151-2157、:2335-2344〕。
- **prefill 也讀量化後的 KV**：KV 先寫進 paged cache（`reshape_and_cache_flash`），attention 再從 cache 讀〔程式碼 flashinfer.py:569、:2532-2541、:2151-2159〕。所以 vLLM 原生 FP8 在**第一輪**就會影響輸出，不是只有還原時才影響〔判讀〕。
- **一個要避開的坑**：checkpoint 若帶 `q_scale ≠ 1`，sm_86 上 Q 沒有被量化（仍是 BF16），FlashInfer 卻照樣把 `sm_scale` 乘上 `q_scale`〔程式碼 vllm flashinfer.py:952-961、:2154；flashinfer prefill.py:2720-2721；compressed_tensors.py:1105-1107、:1209〕。這會讓 logit 整體縮放錯誤〔判讀：程式碼推論，未實測〕。所以要用**不含 q_scale 的 KV-only checkpoint**。
- vLLM 官方部落格的 FP8 準確率問題（128K 撈針 BF16 91%、修正前 FP8 13%、修正後 89%）出在 Hopper 的 FA3「整個 attention 用 FP8 算」的累加精度〔文件 https://vllm.ai/blog/2026-04-22-fp8-kvcache〕。sm_86 的 FlashInfer 路徑是先轉 BF16／FP32 再算，所以那個問題不適用〔判讀〕。

---

## 3. 超出 e4m3 範圍（±448）的值：截在 448，不是 NaN

- 寫入路徑：`reshape_and_cache_flash_kernel` → `CopyWithScaleOp` → `fp8::scaled_convert`〔程式碼 vllm@v0.28.0 csrc/libtorch_stable/cache_kernels.cu:11-13、:189-201、:263-310〕。
- 實際轉換：`__nv_cvt_float_to_fp8(x / scale, __NV_SATFINITE, __NV_E4M3)`〔程式碼 vllm@v0.28.0 csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh:490-503、:558-569〕。
- NVIDIA 文件：`__NV_SATFINITE` 會把超過最大有限值的輸入截成帶正負號的 MAXNORM；`__NV_NOSAT` 才會變 NaN〔文件 https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html〕。
- Triton 的 per-token-head 路徑：明確 `tl.clamp`〔程式碼 triton_reshape_and_cache_flash.py:220〕。Triton 的 per-tensor 路徑靠 `tl.store` 隱式轉型，飽和行為**未查證**（在 sm_86 上本來就跑不了）〔程式碼 同檔 :105-128〕。
- **自己寫 codec 時要注意 torch 版本**：
  - torch 2.13.0：有限值溢位或 inf → 截成最大有限值（0x7e＝448）；NaN → NaN〔程式碼 pytorch@v2.13.0 torch/headeronly/util/Float8_e4m3fn.h:178、:206-213〕。
  - torch 2.9.0：≥480 一律變 NaN（0x7f）〔程式碼 pytorch@v2.9.0 同檔 :177、:205-207〕。
  - 3090 venv 實際是哪個 torch：**未查證**（vLLM 0.28.0 要求 2.13.0）。
- e4m3 的格子：尾數 3 位。同一個 2 的冪區間內，相鄰兩值差 2^(e−3)。在 [256, 448] 之間相鄰差 32，最大捨入誤差 16；最大相對捨入誤差約 3.1–6.25%〔判讀：格式定義推算〕。

---

## 4. Qwen2／Qwen2.5 的 FP8 KV 已知問題

| 來源 | 模型／設定 | 量到什麼 | 可信度 |
|:--|:--|:--|:--|
| 本專案 3090，8/31 | Qwen2.5-7B-Instruct-1M **AWQ 權重**，32,245 token，撈針 20 題 | BF16 20/20、FP8（scale 1.0）1/20、INT8 每 token 每 head 19/20、INT4 0/20 | 〔實測 20260831-181930-m5-needle，D6_precision.md:239 重算過〕 |
| 本專案 MI300X，9/16 | 同模型 **BF16 權重**，要求 131,072 token，撈針 20 題 | bf16 20/20、fp8 2/20、**fp8 每 token 每 head 0/20**、int8 18/20、int4 0/20 | 〔實測 20260916-065920-m5-b-qwen7b-needle；results/m8_directions/d6_quality_inventory.csv:249-253〕。原 CSV 已遺失，只剩 stdout 摘要。fp8_ptk 走 TRITON_ATTN，和 fp8 不同 kernel〔見 01_advisor_feedback.md §3〕 |
| vLLM-ROCm-Windows fork issue #28、PR #30（第三方，2026-09-28） | Qwen2.5-7B-Instruct-GPTQ-Int4，RX 7900 XT／7800 XT | max\|K\| 419–423、max\|V\| 72.5（128／2K／8K token）；(head, channel) 的 max\|K\| 中位數 6.9、p95 93、512 組裡 22 組 >100；集中在 kv head 0、1 的 channel 58–63、123–127；第 27 層 423、第 0 層 173 | 〔文件 https://github.com/ThePie88/vLLM-ROCm-Windows/issues/28〕。非上游、權重是 GPTQ、我沒重現 |
| 同上（第 27 層 attention 誤差，max-rel／rms-rel） | 只量化 K，V 不動 | scale 1.0：0.911／0.237；全範圍 per-tensor scale：0.709／0.362；每 (head, channel) scale：0.937／0.426；每 token scale：0.879／0.444；**減掉每 channel 平均再加每 channel scale：0.031／0.017**；INT8 非對稱每 channel：0.008／0.003 | 〔文件 同上 issue #28〕 |
| 同上 PR #30 的解釋 | — | 「k_proj bias 在最低頻的 RoPE channel 幾乎原封不動」；FP8 是相對精度，所以 ~423 的值絕對誤差可到 ~16；「只改 scale 修不好」；減掉平均是安全的，因為 q·mean 對同一個 query 是常數，softmax 會消掉 | 〔文件 https://github.com/ThePie88/vLLM-ROCm-Windows/pull/30〕 |
| vLLM issue #41343（open） | Qwen2-VL／Qwen2.5-VL，`fp8_e5m2` 不校準 | 靜默產生錯誤輸出；後續留言（vLLM 0.26.0、RTX 5070、Qwen2-VL-2B）說 E5M2 和 E4M3 都錯（白色圖片答成 Blue） | 〔文件 https://github.com/vllm-project/vllm/issues/41343〕。沒有準確率數字 |
| TensorRT-LLM issue #4218 | Qwen2.5-Coder-7B，FP8 權重＋FP8 KV | 輸出亂碼；ModelOpt 警告 KV 有大 activation；轉給 ModelOpt 後關閉，沒有根因 | 〔文件 https://github.com/NVIDIA/TensorRT-LLM/issues/4218〕 |
| SGLang issue #39942（open） | Qwen2／Qwen3-MoE／Mixtral | 載入時 KV scale 名稱對不上被丟掉，靜默退回 1.0 | 〔文件 https://github.com/sgl-project/sglang/issues/39942〕。SGLang，不是 vLLM；vLLM 0.28.0 已有 `qkv_proj`、`qkqkv_proj` 的改名規則〔程式碼 weight_utils.py:1422-1427〕 |
| KVTuner（ICML 2025） | Qwen2.5-7B-Instruct，整數量化 | wikitext PPL：KV8 9.56、K4V8 220.83、KV4 235.03〔原文 §4.3 Table 2〕；GSM8K：KV8 0.7712、KV4 0.0078〔原文 Table 5〕；原文說「sensitive even to int4 key cache quantization」 | 〔原文 arXiv 2502.04420v5，HTML 全文由 WebFetch 摘要〕；沒測 FP8 |
| DiffKV（SOSP 2025） | Qwen2.5-7B | K4V8 時準確率「drops to nearly zero」；因此對 Qwen2.5-7B 關掉低精度 | 〔原文 arXiv 2412.03131v3 §7.2，HTML 由 WebFetch 摘要〕 |

### 4.1 判讀：Qwen2.5 的 FP8 失敗，主因多半不是「scale＝1.0 造成溢位」

1. 外部量到 Qwen2.5-7B-Instruct 的 max\|K\| 約 423，小於 e4m3 的 448〔文件 fork #28〕。若 1M 版本差不多，scale＝1.0 根本不會溢位〔判讀；1M 版本與 BF16 權重的值**未查證**〕。
2. 我們在 MI300X 上的「每 token 每 head 動態 FP8」理論上不會溢位，卻是 0/20〔實測 20260916-065920〕。這格混了 kernel 差異，只能當旁證〔判讀〕。
3. 對浮點格式，per-tensor scale 只是把整個分布搬到別的指數區間；一般範圍內的相對誤差不變。校準只影響兩端：大於 448·s 的被截，小於 2⁻⁶·s 的落入次正規區〔判讀：e4m3 格式定義〕。若 max\|K\|≈423，校準後的 K scale≈423/448≈0.94，和 1.0 幾乎一樣〔判讀〕。
4. V 的 max 約 72.5，校準後 scale≈0.16，能把很小的 V 值移出次正規區，可能有小幫助〔判讀〕。
5. 和 fork 的機制一致的解釋：K 在少數 channel 有很大、幾乎固定的偏移（k_proj bias）。FP8 在 ~400 附近的格子寬 32，捨入誤差蓋過 q·k 裡有用的小差異〔判讀，依 fork PR #30〕。INT8 每 token 每 head 的格子寬約 423/127≈3.3，所以 INT8 能過〔判讀〕。
6. 結論：**「正確校準的全 FP8」在 Qwen2.5 上，很可能還是過不了**。這不是校準做錯，是格式的問題〔判讀〕。這一點要在 3090 上用 §5 的 Step A 先驗證。

---

## 5. llm-compressor 怎麼產生 FP8 KV scale

- **recipe**：`QuantizationModifier(kv_cache_scheme=…)`，`num_bits: 8`、`type: float`、`strategy: tensor` 或 `attn_head`、`dynamic: false`、`symmetric: true`〔程式碼 llm-compressor@0.14.0 examples/quantization_kv_cache/llama3_fp8_kv_example.py:17-43；llama3_fp8_head_kv_example.py:15-20〕。
- **官方範例同時量化權重**：per-tensor 範例把所有 Linear 的權重和輸入都量成 FP8〔程式碼 llama3_fp8_kv_example.py:22-36〕；per-head 範例用 `FP8_DYNAMIC` 權重〔程式碼 llama3_fp8_head_kv_example.py:15-18〕。vLLM 文件的範例則加了一個 `LlamaAttention` 的 q 量化 group〔程式碼 vllm@v0.28.0 docs/features/quantization/quantized_kvcache.md:141-151〕。**這些都會混入權重或 q 的變因。**
- **只量化 KV 是可以的**：沒給 config group 時，llm-compressor 用一個空的預設 scheme（只有 targets，沒有權重／激活量化）〔程式碼 llm-compressor@0.14.0 src/llmcompressor/modifiers/quantization/quantization/mixin.py:373-389〕；有 `kv_cache_scheme` 就一定要校準資料〔程式碼 同檔 :298-305〕。實際的 KV-only checkpoint 存出來是 `config_groups: {}`（見 §5.1）。
- **樣本數**：範例都是 512 筆 × 2,048 token（`perfectblend` 或 `ultrachat_200k`）〔程式碼 llama3_fp8_kv_example.py:46-53；vllm docs quantized_kvcache.md:121-125〕。
- **看的是哪個 K**：hook 掛在 attention 的 KV cache 更新上，收的是 **RoPE 之後**的 key〔程式碼 compressed-tensors@main（2026-10-13）src/compressed_tensors/modeling/kvcache.py:149-165；llm-compressor@0.14.0 mixin.py:499-503、calibration.py:210-211〕。
- **observer 決定 scale 是什麼最大值**：
  - `minmax`：**移動平均**（`averaging_constant` 預設 0.01）〔程式碼 llm-compressor@0.14.0 src/llmcompressor/observers/min_max.py:44-67〕。最大值會被平滑，比真實峰值小〔判讀〕。
  - `static_minmax`：所有樣本的全域 min／max〔程式碼 同檔 :22-41〕。
  - `memoryless_minmax`：只看當次〔程式碼 同檔 :10-19〕。
  - vLLM 寫入時會截在 448，所以 scale 偏小 → 峰值被截；scale 偏大 → 小值掉進次正規區〔判讀〕。
- **README 的一句話**：「fp8 computation is supported on … compute capability > 8.9」〔程式碼 llm-compressor@0.14.0 examples/quantization_kv_cache/README.md:5〕。這是指 FP8 **運算**；FP8 **儲存**在 3090 上可用（第一階段 M1 量到容量 2 倍，見 CLAUDE.md §3）〔判讀〕。

### 5.1 HF 上現成的 checkpoint（2026-10-13 用 HF API 搜尋、逐一讀 config.json）

| repo | KV scale | 權重 | 備註 |
|:--|:--|:--|:--|
| `nm-testing/Llama-3.1-8B-Instruct-KV-Cache-FP8` | per-tensor，`static_minmax` | **BF16**（`config_groups: {}`） | 64 個 scale（32 層 × k/v），沒有 q_scale〔文件 HF config.json、model.safetensors.index.json、recipe.yaml〕。README 的建立程式碼被註解掉，而且寫的是 Qwen3-8B，**校準資料不明** |
| `nm-testing/Llama-3.1-8B-Instruct-KV-FP8-tensor-static_minmax` | per-tensor，`static_minmax` | BF16 | 同上，test 組織〔文件 HF〕 |
| `nm-testing/Llama-3.1-8B-Instruct-KV-FP8-Tensor` | per-tensor，`minmax`（移動平均） | BF16 | 〔文件 HF〕 |
| `nm-testing/Llama-3.1-8B-Instruct-KV-FP8-attn_head-static_minmax` | per-head | BF16 | sm_86 沒有 backend 支援 per-head（§1.4）〔判讀〕 |
| `chengjiyao/Llama-3.1-8B-Instruct-FP8-KV` | per-tensor，`minmax` | BF16（所有 Linear 在 ignore） | 第三方〔文件 HF〕 |
| `amd/Llama-3.1-8B-Instruct-FP8-KV` | `kv_cache_scheme: static` | `quant_method: fp8`、`activation_scheme: static` → 權重也是 FP8 | 會混入權重量化〔文件 HF；判讀〕 |
| `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8`、`…-FP8-dynamic` | **沒有**（`kv_cache_scheme: null`） | FP8 | 只量權重／激活〔文件 HF〕 |
| `RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic` | 沒有 | FP8 | 〔文件 HF〕 |
| `cortecs/…`、`JamAndTeaStudios/…`、`duydq12/Qwen2.5-7B-Instruct-1M-FP8-Dynamic` | 沒有 | FP8 | 〔文件 HF〕 |
| `RedHatAI/Llama-3.1-8B-Instruct-KV-Cache-FP8`（nm-testing README 提到的名字） | — | — | HF API 回「Invalid username or password」，可能不公開或不存在〔未查證〕 |

- **Qwen2.5-7B-Instruct 與 -1M：找不到任何帶 KV scale 的 checkpoint**〔判讀：限 HF 搜尋 `Qwen2.5-7B-Instruct-FP8`、`Qwen2.5-7B-Instruct-1M-FP8`、`FP8-KV` 三組關鍵字的結果〕。最接近的是 `boboliu/Qwen2-7B-Instruct-FP8-KV-CN`，但那是 Qwen2，而且 `quant_method: fp8`（權重也量化）〔文件 HF config.json〕。

---

## 6. 3090 上要怎麼做才算「正確校準」〔判讀〕

以下全部是建議，沒有實測。

**Step A（先做，約 30 分鐘）：量 K／V 的實際分布，決定假設死活。**
- 用 BF16 權重的 Qwen2.5-7B-Instruct-1M，跑 3–5 個 32K prompt，在 RoPE 之後的 K（就是實際存進 cache 的 K）掛 hook。
- 記錄：每層、每 KV head、每 channel 的 max\|K\| 和平均 K；每層的 max\|V\|。
- 判定：若所有層 max\|K\| < 448，「scale＝1.0 溢位」對 3090 就不成立，per-tensor 校準不會改變結果（§4.1 第 3 點）。若某些 channel 平均值很大（例如 >50），就是 fork 說的偏移。
- 8/31 的 1/20 是 AWQ 權重；這次換 BF16 權重，結果可能不同，所以要重量。

**Step B：端點 E1 用自己的 codec，不用 vLLM 原生 FP8。**
- 03 計劃寫的「每 (block, 層, K/V, KV head) 的 amax ÷ 448」〔見 03_plan_h200.md §1〕，在溢位和次正規兩端都比 vLLM 的 per-tensor 靜態 scale 好。用它當「正確校準」，對停損是保守方向。
- 在報告裡分開寫兩件事：(1) scale 怎麼來（每 block 動態，不需要校準資料）；(2) 尾數只有 3 位這件事，**任何 scale 都改不了**。

**Step C：E1 若沒過 ε，在同一個 harness 加兩個診斷（都是換 codec，不是換位置）。**
- C1：K 減掉每 (層, head, channel) 的校準平均，再對殘差做 amax ÷ 448（fork PR #30 的做法）。過了 → 失敗原因是「K 偏移 × FP8 尾數」。
- C2：INT8 每 token 每 head（就是 K3）。
- 兩個任何一個過了，都代表是 codec 的問題，位置策略一樣停（和 03 的 K3 一致）。
- 注意：C1 已經不是「純 e4m3」，多了一個仿射偏移，等於非對稱量化。報告要寫明。

**Step D（選做）：vLLM 原生 FP8 當參考點。**
- 用 KV-only checkpoint：權重 BF16、`config_groups: {}`、per-tensor、`static_minmax`、**不帶 q_scale**（§2.1 的坑）。
- Llama 可以直接用 `nm-testing/Llama-3.1-8B-Instruct-KV-Cache-FP8`，但校準資料不明；Qwen 要自己做。
- 校準長度要接近 32K。fork 量到 Qwen2.5-7B 在 128／2K／8K 的 max\|K\| 幾乎一樣〔文件 fork #28〕，32K 是否一樣**未查證**。
- 啟動參數：BF16 和 FP8 都加 `--attention-backend FLASHINFER`；FP8 加 `--kv-cache-dtype fp8`。
- 檢查：server log 有「Using FLASHINFER attention backend」；每層的 `_k_scale_float`、`_v_scale_float` 不是 1.0（SGLang 有靜默丟 scale 的前例）。
- 不要用 `attn_head` checkpoint（sm_86 沒有 backend），不要用 `fp8_per_token_head`（sm_86 上能不能跑**未查證**，要先 smoke test）。

**Step E：codec 的單元測試再加兩條。**
- 印出 `torch.__version__`。2.13.0 會飽和、2.9.0 會變 NaN（§3）。
- 編碼後檢查沒有 NaN／Inf，並記錄被截在 ±448 的元素比例（用 amax ÷ 448 時應該是 0）。

---

## 7. 沒查到、要留意的

- 3090 venv 裡的 FlashInfer 是否真的被選到（8 月的 server log 沒重看）〔未查證〕。
- `fp8_per_token_head` 在 sm_86 的行為〔未查證〕。
- Qwen2.5-7B-Instruct-**1M** 的 K 分布是否和 7B-Instruct 一樣有大偏移〔未查證〕。
- FlashInfer decode 在 GQA 比例 7（Qwen2.5-7B：28 個 Q head、4 個 KV head）時是否改走 tensor-core 版 kernel；不論哪條，K/V 都先轉成 BF16 或 FP32 再算〔判讀〕。
- `calculate_kv_scales` 的第一個 batch 是不是 profiling 假資料〔未查證〕。現在已移除，不影響 0.28.0。
