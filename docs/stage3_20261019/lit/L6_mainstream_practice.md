# L6　業界實際怎麼處理 KV：引擎預設、業者揭露、架構趨勢，與「精度能降到多低」的正式證據

> **日期**：2026-10-20（UTC 04–05 時）
> **執行者**：文獻 agent。只讀檔、只上網、只用 CPU；**沒有開任何 GPU 行程，沒有 import torch／vllm／transformers**（套件的原始碼只用 grep／sed 讀文字檔）；本機指令都用 `nice -n 19 taskset -c 0-3`。
> **要回答的問題**（使用者原話）：「目前大家推論的主流都用什麼？有 paper 或正式資料嗎？不然精度來講會不會越低越好？」
> **和 L0–L5 的分工**：學術的壓縮方法（L2）、低秩（L5a）、合併／跨層（L5b）、卸載時壓縮（L5c）都已經有卡片，這裡不重做。本檔只補兩件事：**業界實際在用什麼**，以及**有沒有正式資料說明精度能降到多低**。
> 本檔是 **self-review**，不是 cross-model review（CLAUDE.md §5）。

**標記**
- 來源性質：〔官方文件〕＝引擎或廠商的正式文件；〔程式碼〕＝原始碼，附 `檔案:行號`；論文、部落格直接寫在出處欄。
- 閱讀深度：〔全文〕＝整篇讀過；〔部分〕＝用 WebFetch 依我指定的欄位抽取（由工具轉述，**不是逐句人工核對**），或下載到本機後只讀相關段落；〔摘要〕＝只讀摘要頁或搜尋結果摘要。標「逐字」的引文是在本機下載的原始檔裡直接看到的，可以 grep 重現（存檔位置見 §9）。
- 句子層級：〔判讀〕＝我對來源的解讀；〔推論〕＝來源沒寫、我推出來的；〔算術〕＝我用公開設定檔或公開數字算的；〔實測 run〕＝本專案量到的，附 run_id。**NOT_FOUND**＝查了沒找到（≠不存在，CLAUDE.md 規則 7）；**NOT_VERIFIED**＝有線索但沒核到原始來源；**NOT_MEASURED**＝本專案沒量。
- **廠商宣稱，非獨立驗證**：NVIDIA、AMD、Google、vLLM／SGLang 團隊、模型業者對自家技術的精度說法，一律加這個標記。
- 不同硬體、模型、題目、context 的數字**只並列，不排名**。

---

## 0. 一頁結論

1. **業界主流是「先用架構把 KV 縮小，再把 KV 存成 FP8」；但所有引擎的「預設」仍然是 16 位元。**
   - **預設**：vLLM、SGLang 都是 `auto`（跟模型 dtype，我們兩個模型都是 BF16）；llama.cpp K、V 都是 `f16`；TensorRT-LLM 跟 checkpoint 走〔程式碼／官方文件，§2〕。例外只有少數模型在程式碼裡被強制或自動改成 FP8：DeepSeek-V3.2 在 vLLM 0.28 的 NVIDIA 路徑**必須**用 FP8 KV；SGLang 在 SM100 以上對 DeepSeek DSA 自動選 `fp8_e4m3`〔程式碼〕。
   - **實際開的**：FP8。NVIDIA 寫「FP8 KV caches are well utilized in production」〔廠商宣稱，非獨立驗證〕；vLLM 團隊兩篇部落格把 FP8 當「許多長上下文部署的預設起點」「預設選擇」；SGLang 文件推薦 `fp8_e4m3`；AMD ROCm 文件把 FP8 KV 標成「All inference workloads」；MLPerf 提交裡 NVIDIA（Llama2-70B v4.1、DeepSeek-R1 v5.1）和 AMD（MI300X，v4.1）都用 FP8 KV。業者自己揭露的：Character.AI 用 **int8** KV（2024，而且是原生 int8 訓練）；DeepSeek-V2 平均 **6 bit**、V3.2 **FP8**、V4.1 **FP8＋FP4**（§3）。
   - **沒有任何公開的「多少比例的部署開了 FP8 KV」統計**（NOT_FOUND）。OpenAI、Google（Gemini）、Meta、Moonshot、阿里的**生產環境 KV 精度都沒有公開**（NOT_FOUND）；它們公開的是架構（§3、§4）。
   - **正式資料有，但分三級**：MLPerf（最正式；但有用 FP8 KV 的 Llama2-70B 題目 `max_seq_len=1024`，說明不了 32K 以上）＞同儕審查論文（Li et al. ICML'24、KVQuant NeurIPS'24 等，多在 32K 以內）＞廠商與專案部落格（數字多、條件窄、自測）。§5 逐條標級。

2. **精度不是越低越好**。三組證據方向一致：
   - **品質有懸崖，而且在長 context 往往更嚴重**：8 bit 幾乎無損；做得好的 4 bit（K 逐 channel 分組的 KIVI 類）多半無損；**2–3 bit 是懸崖**。KVQuant 在 RULER 32K：fp16 56.40 → 3 bit 53.65、2 bit 36.54、KIVI-2 39.78〔部分〕；D-Quant 表 3：Llama-3.1-8B 的 KIVI（2.25 bit）比 BF16 掉 14.1（32K）→ 24.6（128K）個百分點，Qwen3-8B 掉 21.9 → 38.3〔部分，預印本〕；vLLM 的 TurboQuant 研究：3 bit 的變體在推理題最多掉約 20 分、MRCR 256K 掉 12–15 分〔部分，Red Hat AI〕。我們自己：INT2 KIVI 在 Llama 128K 掉 7.2 [−13.5, −1.4]、Qwen 32K 掉 12.9 [−22.1, −4.9]〔實測 run，§5.4〕。
   - **懸崖在哪裡，「格式×模型的離群值」比位元數更重要**：Qwen2 家族（沒有 QK-Norm）用逐 tensor 的 FP8 就崩——Mittal 等人在 Hugging Face 部落格（Intel Arc Pro B70 上量測）量到 DeepSeek-R1-Distill-Qwen-7B 的 RULER 4K 從 0.5291 掉到 0.0891〔部分〕；我們的 Qwen2.5-7B-1M 用逐 head 的 FP8 掉 37.9（32K）到 55.1（128K）個百分點，**同一個模型用 4 bit 的 INT4 KIVI 卻不掉**〔實測 run〕。8 bit 的 FP8 比 4 bit 的 KIVI 差，說明「位元越少越差」本身也不成立。
   - **位元越低，額外成本越高**：vLLM 在 H100 量到 FP8 的吞吐和 BF16 一樣，TurboQuant 只剩 BF16 的 66–80%〔部分〕；我們在 MI300X 上，vLLM 原生 FP8 讓 decode 每 token 快 4–29%，但我們自寫、把 INT4 反量化融進 attention 的 Triton kernel 每 token 反而變慢（RUNLOG 14.4）〔實測 run〕。
3. **主流停在 FP8 的理由**（§6.3）：硬體有原生格式（Hopper／Ada／Blackwell／MI300 都有；TensorRT-LLM 從 Ampere 起就支援 FP8 KV 儲存）、容量剛好 2 倍、attention kernel 現成而且不會變慢、一般的 GQA 模型幾乎無損。代價是要**校準**（vLLM 不校準時 scale＝1.0），而且有已知的例外模型。**4 bit 是下一步，但有條件**：NVFP4 KV 只有 Blackwell SM100／SM103 有（TensorRT-LLM、SGLang 都標實驗性）；DeepSeek-V4.1 只把「壓縮注意力」那部分 KV 放 FP4，滑動視窗那部分仍是 FP8〔官方文件〕；**MI300X 沒有原生 FP4，vLLM 0.28 的 `nvfp4` 只有 FlashInfer 後端，ROCm 上用不了**〔程式碼〕。
4. **「主流」其實更大的一塊是架構**：用公開 config 算〔算術〕，GQA（Llama-3.1-8B）比 MHA 小 4 倍；MLA（DeepSeek-V3）比同規格 MHA 小 71 倍，而且 70,272 B／token 和 DeepSeek 在 ISCA'25 論文報的 70.272 KB 完全對上；Gemma 3（5 局部:1 全域、視窗 1024）在 128K 小 6.0 倍；Llama 4 Scout（每 4 層 1 層全域）小 3.4 倍；gpt-oss（窗 128 與全域交替）小 2.0 倍；Qwen3-Next／Qwen3.5（3 層線性:1 層全域）小 4 倍；MiniMax-01（7:1）小 8 倍；Kimi Linear 小 74%。**反例**：MiniMax-M2 回到全注意力，理由之一是線性注意力「far more sensitive to numerical precision」〔部分〕。量化是疊在這之上的第二層。我們用的兩個 7–8B dense GQA 模型（Llama 128 KiB／token、Qwen 56 KiB／token）屬於「KV 偏大」的舊設計。
5. **MLPerf 的規則**〔官方文件，逐字〕：KV「should be handled in the same way as the activations」，可以照量化規則量化，**但不可以 prune 或 sparsify**——也就是 closed division 不允許丟 token。門檻是參考實作的 99%（多數 LLM 題）或 99.9%（Llama2-70B）。〔算術〕在我們 Llama 32K 的基線 68.2% 上，99% 等於只能掉 0.68 個百分點；我們的 ε（CI 下界 > −2 個百分點）約等於 97.1%，**比 MLPerf 寬**。
6. **對我們的對照組**（§7）：**主對照組用 BF16（vLLM `auto`）**——它是 vLLM 的預設，也是兩個模型的原生 dtype。**建議加一個「常見部署」的副對照組：Llama 用 vLLM 原生 FP8 KV**。**Qwen2.5-7B-1M 不要用 vLLM 的 `fp8` 當「部署預設」**：在 MI300X 上它是 per-tensor、scale＝1.0、e4m3fnuz（最大值 240），而 Qwen 的 max|K| 約 422；九月實測撈針 2/20，同一批 `int8_per_token_head` 是 18/20。拿已知會壞的設定當對照組，審稿人會說是稻草人。Qwen 改用 `int8_per_token_head`，或明寫「FP8 在此模型失效」。
7. **MI300X＋vLLM 0.28 上 FP8 KV 可用**（RUNLOG 有紀錄；`environment_snapshot.md` 沒有 FP8 這一項＝NOT_FOUND）：
   - 走 ROCM_ATTN，容量剛好 2.000 倍（Llama，154.73 GiB：1,267,504 → 2,535,008 token）。
   - decode 每 token 快 4–29%（Llama 32K −10%、128K −29%；Qwen −4%、−14%），prefill＋1 慢 2–9%（RUNLOG 14.4(d)）。
   - 要注意兩點：AMD 文件寫的 `--calculate-kv-scales` 在 0.28 **已經移除**（PR #49389）；接 KV connector 時 vLLM 會停用 ROCM_ATTN、改走 TRITON_ATTN，這條路徑上 FP8 KV 的速度與品質都是 **NOT_MEASURED**。
8. **哪些是正式的、哪些只是宣稱**：§6.4 的表逐條列。一句話：「FP8 在一般 GQA 模型上近乎無損」有多個獨立來源；「NVFP4／TurboQuant 4 bit 損失 <1%」目前主要是**廠商或作者自測**，獨立評測（D-Quant、vLLM 自己的 TurboQuant 研究）在 2–3 bit 都看到明顯下滑。

---

## 1. 總表：引擎／業者 × KV 作法

> 「預設」和「推薦／實際作法」分開寫。出處編號見 §8。

| 對象 | 架構層的 KV 縮減 | **預設** KV 精度 | 可選／實際用的低精度 | **官方推薦**或業者揭露 | 精度說法（性質） | 出處 |
|:--|:--|:--|:--|:--|:--|:--|
| **vLLM 0.28.0**（我們用的版本） | — | `auto`＝模型 dtype（Llama、Qwen 都是 BF16）〔程式碼〕 | `fp8`／`fp8_e4m3`／`fp8_e5m2`；`fp8_per_token_head`／`int8_per_token_head`／`int4_per_token_head`（只有 TRITON_ATTN）；`nvfp4`（只有 FlashInfer）；`turboquant_*` 四種；`fp8_ds_mla` | 文件：開 FP8 的話「[Recommended]」用 llm-compressor 校準；不校準時 scale 全部＝1.0 | 文件**沒有任何精度數字**；啟動時只印「may cause accuracy drop without a proper scaling factor」〔程式碼〕 | R1–R3 |
| **vLLM 團隊部落格**（2026-04、2026-05） | — | — | FP8；TurboQuant | 「FP8 KV-cache is ready to be the default starting point for many long-context vLLM deployments」；TurboQuant 研究：FP8 是預設選擇，GPU 記憶體不是瓶頸時用 BF16 | 有實驗（H100／B200／H200；MRCR 到 1M、AIME 等）；作者是 Red Hat AI、AWS；非同儕審查 | R5、R6 |
| **SGLang**（main，2026-10-20） | — | `auto`；DeepSeek DSA 在 SM100+ 自動 `fp8_e4m3`；另一個模型（Muse Spark）`auto`→`fp8_e4m3`〔程式碼〕 | `fp8_e4m3`、`fp8_e5m2`、`mxfp8`、`nvfp4`、`fp4_mx_block16` | 「Use fp8_e4m3 for better accuracy (recommended)」；FP4 實驗性 | 文件只有定性說法：FP8「typically minimal」；FP4 在小模型「more pronounced accuracy drops」 | R7–R9 |
| **TensorRT-LLM**（main） | — | 跟 checkpoint〔判讀：checkpoint 沒有 KV 量化設定就不量化〕 | FP8（Ampere 起）；NVFP4（只有 SM100／103，要 ModelOpt 先量化）；程式碼的 KV 量化清單還有 INT8；另有只在 Host／Disk 用的 cold-page NVFP4 | 文件只說「可以手動開 FP8」，沒有推薦語 | 文件**沒有精度數字**；KV 壓縮頁寫「must be validated before deployment」 | R10 |
| **NVIDIA**（部落格） | — | — | NVFP4 KV（Blackwell） | 「FP8 KV caches are well utilized in production」 | NVFP4 對 BF16／FP8「accuracy loss of less than 1%」（Qwen3-480B，含 RULER 64K）〔廠商宣稱，非獨立驗證〕 | R11 |
| **llama.cpp**（master） | — | K、V 都是 `f16`〔官方文件、程式碼〕 | `q8_0`、`q5_0/1`、`q4_0/1`、`iq4_nl`、`bf16`；V 要量化必須開 Flash Attention〔程式碼〕 | 沒有官方推薦 | PR #7412 的 PPL／KLD 表：`q8_0` 幾乎無損；K 比 V 敏感（模型、長度沒寫） | R12、R13 |
| **LMCache** | — | 不設 serde＝存原始位元組〔官方文件〕 | serde `fp8`（e4m3fn）、`turboquant`（預設 preset `turboquant_k8v4`） | 每個 L2 adapter 自己 opt-in | 文件：fp8「minor accuracy loss」，沒有數字 | R14 |
| **NVIDIA Dynamo KVBM** | — | 引擎給什麼就存什麼〔判讀〕 | 設計文件**沒提**量化或壓縮（NOT_MENTIONED） | — | — | R15 |
| **AMD ROCm 文件** | — | vLLM `auto` | FP8（e4m3、e5m2）；AMD 發布的 `*-FP8-KV` 預量化模型 | 推薦開 `--kv-cache-dtype fp8`（表上「All inference workloads」）；最好用 LLM Compressor 校準；**範例還用 `--calculate-kv-scales`，vLLM 0.28 已移除** | 「minimal accuracy impact」沒有數據〔廠商宣稱，非獨立驗證〕；`amd/Llama-3.1-8B-Instruct-FP8-KV` 模型卡：WikiText-2 PPL 7.2169 → 7.2752（權重＋activation＋KV 一起 FP8） | R16–R18 |
| **AMD**（ROCm 部落格，MI355X） | — | — | LMCache 裡 4 bit TurboQuant（頭尾各 2 層留 BF16） | — | RULER 16K–128K 和 BF16 差約 1 分以內（MiniMax-M2.5）〔廠商宣稱，非獨立驗證〕 | R19 |
| **MLPerf Inference**（規則） | — | 參考實作 FP32／FP16／BF16 | KV 照 activation 的量化規則；**不可 prune／sparsify** | — | 門檻 99% 或 99.9%（相對參考實作） | M1 |
| **MLPerf 提交** | — | — | NVIDIA Llama2-70B v4.1：`--kv_cache_dtype fp8`（OpenOrca 校準 1,024 筆）；NVIDIA DeepSeek-R1 v5.1：KV 從 BF16 量化成 FP8；AMD MI300X v4.1：「implemented FP8 KV cache」 | — | 都過了門檻；但 Llama2-70B 的 `max_seq_len=1024` | M2–M5 |
| **Character.AI**（2024-06） | MQA（比 GQA 小 8 倍）；局部注意力視窗 1024，每 6 層只有 1 層全域；相鄰層共用 KV（再小 2–3 倍）；合計 >20 倍 | NOT_FOUND | **int8** KV（權重、activation 也是 int8），**原生 int8 訓練** | 業者揭露〔全文〕 | 「without regressing quality」，沒有數字〔業者宣稱〕 | O1 |
| **DeepSeek** | MLA（V2 比 67B 少 93.3%；V3 是 70,272 B／token） | NOT_FOUND（V3 線上服務只寫「core MLA computations … use the BF16 format」，指計算不是儲存） | V2：KV 平均 6 bit；V3.2：FP8（NoPE 512 維 FP8，RoPE 64 維「not quantized for accuracy」）；V4.1：FP8 528 B／token，壓縮注意力 FP4 288 B／token；KV 存磁碟陣列（Context Caching on Disk、3FS） | 業者揭露 | 沒有 KV 量化的精度數字 | O2–O7 |
| **Moonshot／Kimi** | K2 用 MLA（config 和 DeepSeek-V3 同型）；Kimi Linear：KDA:MLA＝20:7，KV「up to 75%」更小 | NOT_FOUND | Mooncake：CPU DRAM／SSD 組 KV 池 | 業者揭露（架構與系統） | KV 精度 NOT_FOUND | O8、O9 |
| **Google** | Gemma 3：5 局部:1 全域、局部視窗 1024 | Gemini 生產 NOT_FOUND | TurboQuant（研究，3–3.5 bit） | 部落格沒說 Gemini 有用 | 作者：3.5 bit「absolute quality neutrality」〔作者宣稱〕 | O10、O16 |
| **OpenAI** | gpt-oss：窗 128 的局部注意力與全域交替、64 Q／8 KV heads、attention sink | NOT_FOUND | NOT_FOUND | — | — | O11 |
| **Meta** | Llama 4：iRoPE；config 每 4 層 1 層全域（NoPE），其餘是 8,192 的 chunked attention | NOT_FOUND | NOT_FOUND | — | — | O12 |
| **阿里 Qwen** | Qwen3 加 QK-Norm；Qwen3-Next／Qwen3.5：Gated DeltaNet:Gated Attention＝3:1，2 個 KV head | NOT_FOUND（生產） | 和 SGLang、NVIDIA 合寫 NVFP4 KV 評測（實驗性、scale＝1.0） | — | 對 FP8：GSM8K −0.08～−0.31、SWE-bench Verified −1.6 分〔廠商宣稱，非獨立驗證〕 | O13、O14、R9 |
| **MiniMax** | 01：lightning:softmax＝7:1；**M2 回到全注意力** | NOT_FOUND | — | M2：線性注意力「far more sensitive to numerical precision」 | — | O15 |

---

## 2. 推論引擎：「預設」和「推薦」分開看

### 2.1 vLLM（0.28.0，我們 MI300X 上用的版本；commit 2cf0a69，2026-08-24）

- **預設**：`cache_dtype: CacheDType = "auto"`，docstring「If "auto", will use model data type」〔程式碼 `vllm/config/cache.py:77-78`，逐字〕。我們兩個模型的 config 都是 `torch_dtype: bfloat16`，所以預設就是 BF16 KV〔程式碼，本機 config〕。
- **可選值**（17 種）：`auto, float16, bfloat16, fp8, fp8_e4m3, fp8_e5m2, fp8_inc, fp8_ds_mla, turboquant_k8v4, turboquant_4bit_nc, turboquant_k3v4_nc, turboquant_3bit_nc, int4_per_token_head, int8_per_token_head, fp8_per_token_head, nvfp4, nvfp4_4over6`〔程式碼 `cache.py:19-37`〕。
  - docstring 寫「ROCm (AMD GPU) supports fp8 (=fp8_e4m3)」，也寫「Some models (namely DeepSeekV3.2) default to fp8」〔程式碼 `cache.py:79-82`，逐字〕。
- **哪個後端支援哪些格式**〔程式碼〕：
  - ROCM_ATTN、ROCM_AITER_FA、ROCM_AITER_UNIFIED_ATTN：auto／fp16／bf16／fp8／fp8_e4m3／fp8_e5m2（`rocm_attn.py:171`、`rocm_aiter_fa.py:720`、`rocm_aiter_unified_attn.py:31`）。
  - TRITON_ATTN：再加三種 `*_per_token_head`（`triton_attn.py:296-306`）。SM89 以下拒收 fp8 的檢查只在 `is_cuda()` 分支裡（`:536-549`），ROCm 不受影響。
  - `nvfp4`／`nvfp4_4over6` 只在 FlashInfer（`flashinfer.py:408-417`）→ **ROCm 上沒有 NVFP4 KV**。
  - TURBOQUANT 後端有列在 ROCm 的候選清單裡（`platforms/rocm.py:493`），但在 MI300X 上能不能跑：NOT_VERIFIED（這輪不能開 GPU）。
  - **接 KV connector 時不用 ROCM_ATTN**：註解「Keep ROCM_ATTN disabled for KV connectors until connector transfer semantics are validated」（`platforms/rocm.py:482-485`）。這和 `environment_snapshot.md` 記的「接 connector 時退回 TRITON_ATTN」一致。
- **scale 從哪來**〔程式碼 `quantization/kv_cache.py:95-151`〕：checkpoint 有 k／v scale 就用；在 fnuz 平台（MI300 系列，`is_fp8_fnuz()` 判斷 `gfx94`，`platforms/rocm.py:976-978`）會**乘 2**。兩個都沒有就**一律 1.0**，並警告「Using KV cache scaling factor 1.0 for fp8_e4m3」。
  - 〔判讀〕在 MI300X 上不校準時，FP8 KV 是 e4m3fnuz（最大值 240；OCP e4m3fn 是 448，見 R20），scale＝1.0，所以 |K| > 240 的值會被截或溢位（ROCm kernel 是截還是溢位：NOT_VERIFIED）。Qwen2.5-7B 的 max|K| 約 422（本專案 V2 核對，RUNLOG:321）。
- **`--calculate-kv-scales` 已移除**：PR #49389，2026-08-03 合併；v0.28.0 比它新 580 個 commit（本專案 P04 卡已核）。
- **官方文件**（0.28 本機版與線上 latest，頁尾 2026-08-03）〔官方文件，部分〕：
  - 把 FP8 KV 描述成增加容量、提高吞吐的手段。
  - 校準有三種，「[Recommended] Calibration with a dataset (via llm-compressor)」；不校準時「All quantization scales are set to 1.0」。
  - **整頁沒有任何精度數字**。
  - 有 `--kv-cache-dtype-skip-layers`：可以把滑動視窗層等「more sensitive to KV-cache quantization」的層留在原生 dtype。
- **模型層強制 FP8**〔程式碼〕：
  - DeepSeek-V3.2（NVIDIA 路徑）`require_fp8_kv_cache = True`，不是 FP8 就 assert 失敗（`models/deepseek_v32/attention.py:162, 273-285`）。
  - DeepSeek-V4 用 `fp8_ds_mla` 版面時，非 FP8 直接報錯，FP8 則改寫成 `fp8_ds_mla`（`models/deepseek_v4/attention.py:104-114`）。
  - 稀疏 MLA 後端會把 `auto` 或 FP8 正規化成 `fp8_ds_mla`（`model_executor/layers/attention/mla_attention.py:349-361`）。
- **vLLM 團隊部落格 1：「The State of FP8 KV-Cache and Attention Quantization in vLLM」**（2026-04-22；Red Hat AI＋AWS）〔部分〕
  - 所有評測都用不校準的 per-tensor scale（＝1.0）。
  - 128K 撈針在 Hopper／FA3 上 BF16 91% → FP8 13%，原因是「imprecise FP32 accumulation in the Tensor Cores」；改成兩段累加後回到 89%。〔判讀〕**FP8 KV 的品質也取決於 attention kernel**，不只取決於儲存格式。
  - 其他結果：推理題（Qwen3-30B-A3B-Thinking、Qwen3.5-27B）最多掉 1–2、0.7 分；MRCR：Llama-3.3-70B 在 128K 保留 97–98% 的 AUC，Qwen3-30B-A3B 在 256K 約 94%。Kimi-K2.5（FlashMLA，不校準）各長度「consistently negative」，建議校準。
  - 結論：FP8「ready to be the default starting point for many long-context vLLM deployments」。以下情況建議留在 BF16：context <7K、`head_dim=256` 而且在乎 prefill（TTFT 約 1.6 倍）、不校準時答對率掉到 95% 以下、模型有很多小的滑動視窗層。
  - **沒有 AMD／ROCm 的數據**。
- **vLLM 團隊部落格 2：「A First Comprehensive Study of TurboQuant」**（2026-05-11；Red Hat AI；全部在 H100）〔部分〕
  - 容量：FP8 2 倍、k8v4 2.4 倍、4bit-nc 最多 3.4 倍。
  - 吞吐（相對 BF16）：FP8 一樣；TurboQuant 在 Qwen3-30B 是 73–80%，在 Llama-70B 是 66–75%。
  - MRCR（Qwen3-30B-A3B，到 256K）：BF16 45.8、FP8 43.1、k8v4 43.0、4bit-nc 42.3、k3v4-nc 33.5、3bit-nc 31.2。推理題：3 bit 的兩種變體掉約 20 分。
  - 建議：「FP8 is the default choice」；4bit-nc 只在記憶體吃緊、驗證過才用；3 bit 不建議上線；記憶體不是瓶頸時用 BF16。

### 2.2 SGLang（main，2026-10-20 下載）

- **預設**：`kv_cache_dtype` 預設 `"auto"`，「"auto" will use model data type」；可選 `auto, fp8_e5m2, fp8_e4m3, mxfp8, bf16, bfloat16, nvfp4, fp4_mx_block16`〔程式碼 `python/sglang/srt/arg_groups/fields/model.py:197-221`，逐字〕。
- **自動改成 FP8 的情況**〔程式碼 `arg_groups/overrides.py`〕：
  - DeepSeek DSA 在 compute capability ≥10 而且沒有 learnable sink 時，`auto` → `fp8_e4m3`，否則 BF16。警告文字「DeepSeek V3.2 defaults to FP8 KV cache」（`:490-535`）。
  - 名為 Muse Spark 的模型 `auto` → `fp8_e4m3`（`:309-321`；開發者未查證）。
- **官方文件**〔官方文件，部分〕：
  - 「Use `fp8_e4m3` for better accuracy (recommended)」。
  - FP8 E4M3「typically introduces minimal accuracy degradation」；FP4「currently experimental」，「On smaller models, FP4 shows more pronounced accuracy drops」。
  - 只支援 per-tensor scale；checkpoint 沒有就「default to 1.0, which may cause accuracy issues」。
- **LMSYS 部落格「Accelerating Long-Context and Agentic Inference with NVFP4 KV Cache」**（2026-09-16；SGLang、Qwen、NVIDIA 三方合寫）〔部分〕
  - 只和 FP8 比，沒有 BF16 基線。
  - Qwen3.5-397B：GSM8K 少對 1 題（約 −0.08）；Qwen3.8-27B：GSM8K −0.31、GPQA −1.01、SWE-bench Verified 76.20% vs 77.80%（−1.60）。
  - global scale 用 1.0、沒有校準；標「experimental」；只支援 GQA 和 Sparse MLA；硬體是 Blackwell。〔廠商宣稱，非獨立驗證〕

### 2.3 TensorRT-LLM（main，2026-10-20 下載）

- 文件列出的 KV 量化只有「FP8 KV Cache」「NVFP4 KV Cache」〔官方文件 `docs/source/features/quantization.md:13-14`〕：
  - FP8「allows you to enable the FP8 KV cache manually, even for checkpoints that do not have it enabled by default」（`KvCacheConfig(dtype='fp8')`）。
  - NVFP4「offline quantization with ModelOpt is required」，而且目前只能搭配 FP8 權重／activation（`:51-91`）。
- 硬體矩陣〔官方文件 `:125-133`，逐字表〕：FP8 KV＝Blackwell sm120／sm100／103、Rubin sm107、Hopper、Ada、**Ampere** 都是 Y；NVFP4 KV＝**只有 sm100／103**。
- 程式碼的 KV 量化清單：`KV_CACHE_QUANT_ALGO_LIST = [QuantAlgo.FP8, QuantAlgo.INT8, QuantAlgo.NVFP4]`〔程式碼 `tensorrt_llm/quantization/mode.py:55`〕。INT8 KV 還在程式碼裡，但現行文件沒列。
- 另有 cold-page 量化：GPU 上維持原本 dtype，**只在 page 搬到 Host 或 Disk 時存成 NVFP4**〔官方文件 `quantization.md:64-66`、`kv-cache-compression.md`；L2／L5c 已整理〕。文件寫「accuracy and quality impact must be evaluated for the target workload」。
- **文件沒有任何 KV 量化的精度數字**。

### 2.4 llama.cpp（master，2026-10-20 下載）

- **預設**：`-ctk, --cache-type-k`、`-ctv, --cache-type-v`「(default: f16)」；可選 `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`〔官方文件 `tools/server/README.md`，部分〕；程式碼預設 `type_v = GGML_TYPE_F16`（`src/llama-context.cpp:4003`）。
- **V 量化要 Flash Attention**：「quantized V cache was requested, but this requires Flash Attention」（`:558-559`）；`-fa auto` 時會自動打開（`:4063-4069`）〔程式碼，逐字〕。
- **唯一找到的官方倉庫內數據**：PR #7412（J. Gäßler，2024-05-20，研究用 demo，後來被 #7527 取代）〔部分〕。PPL／KLD（模型、context 沒寫）：
  - f16／f16：PPL 6.2322，KLD 0.00019
  - q8_0：PPL 6.2344，KLD 0.00098
  - q5_0：PPL 6.2789
  - q4_1：PPL 6.3425
  - q4_0：PPL 6.4384，KLD 0.0365
  - 作者結論：「K cache is much more sensitive」，q8_0「no significant quality loss」。
- **沒有官方推薦**（NOT_FOUND）。

### 2.5 LMCache、NVIDIA Dynamo

- **LMCache**〔官方文件，部分〕：「Serde is opt-in per adapter: one `--l2-adapter` may use fp8 while another stores raw bytes」；不設 serde「behaves exactly as if serde did not exist」。
  - 內建 `fp8`（預設 `float8_e4m3fn`），文件寫「halves byte volume vs. bf16 with minor accuracy loss」（沒有數字）。
  - 內建 `turboquant`（預設 preset `turboquant_k8v4`）、`aesgcm`（加密）。
  - CacheGen 只被當成「可以自己寫的 serializer」的例子。
- **NVIDIA Dynamo KVBM**（v1.3.0 設計文件）〔官方文件，部分〕：G1 GPU、G2 CPU pinned、G3 本地 SSD、G4 遠端。只有 layout metadata 有 `dtype` 欄；**沒有提到任何量化或壓縮**（NOT_MENTIONED）。〔判讀〕KVBM 搬的是引擎給的原始 KV。

### 2.6 AMD ROCm（MI300X 的官方建議）

- **ROCm 文件「vLLM V1 performance optimization」**（latest，2026-10-20 下載）〔官方文件，逐字〕：
  - 量化方法表把「FP8 KV-cache｜8-bit float｜Excellent｜KV cache: 50%｜**All inference workloads**」列為一列。
  - 「Use `--kv-cache-dtype fp8` to enable FP8 KV-cache quantization. For best accuracy, use calibrated scaling factors generated via LLM Compressor. Without calibration, scales are calculated dynamically (`--calculate-kv-scales`) with minimal accuracy impact.」
  - 格式選項：「fp8 or fp8_e4m3: Higher precision (default, recommended)」。
  - 列出 AMD 發布的預校準模型：`amd/Llama-3.1-8B-Instruct-FP8-KV`、`amd/Llama-3.3-70B-Instruct-FP8-KV`。
- **文件和程式碼對不上**〔推論〕：`--calculate-kv-scales` 在 vLLM 0.28 已移除（§2.1）。照 AMD 文件在 0.28 上下指令，這個旗標應該會被拒收（沒有實際試，NOT_VERIFIED）。「minimal accuracy impact」沒有附任何數據〔廠商宣稱，非獨立驗證〕。
- **`amd/Llama-3.1-8B-Instruct-FP8-KV` 模型卡**〔部分〕：權重、activation、KV 都是 FP8 per-tensor（Quark，用 Pile 校準）；WikiText-2 PPL 7.2169 → 7.2752；卡上寫「pseudo-quantization mode」「for reference only」；沒有長上下文評測。
- **ROCm 部落格「Enhancing vLLM Inference on AMD GPUs」**（2024-09-19，2025-06-09 更新）〔部分〕：「With ROCm 6.2, the KV cache can now be stored in FP8 format」，「effectively enables you to double the sequence length or batch size」；**沒有精度說明**。
- **ROCm 部落格「4-bit KV Caching in LMCache … on AMD MI355X」**（2026-08-28）〔部分〕：
  - TurboQuant 4 bit，頭尾各 2 層留 BF16；MiniMax-M2.5；TP＝2。
  - 對 BF16「within ~1 pt」；RULER 16K–128K 從 DRAM 重載的差 −0.1～+0.4。
  - 等記憶體下完成請求 385 vs 189、p95 TTFT 53 vs 140 s。
  - **只有 MI355X**，沒有 MI300X 的數字。〔廠商宣稱，非獨立驗證〕

---

## 3. 業者公開揭露的實際作法

### 3.1 Character.AI（「Optimizing AI Inference at Character.AI」，2024-06-20）

〔全文〕原址 `research.character.ai/optimizing-inference/` 現在轉址到 blog 首頁。這裡讀的是 Wayback Machine 存檔（本機副本見 §9）；`blog.character.ai` 上的公告版只有摘要，沒有技術細節。逐字重點：

- 「We use the following techniques to reduce KV cache size by more than 20X without regressing quality.」
- **MQA**：「We adopt Multi-Query Attention … in all attention layers. This reduces KV cache size by 8X compared to the Grouped-Query Attention adopted in most open source models.」
- **局部／全域混合**：「reducing attention horizon to 1024 on most attention layers does not have a significant impact on evaluation metrics, including the long context needle-in-haystack benchmark. In our production model, only 1 out of every 6 layers uses global attention.」
- **跨層共用 KV**：「We tie the KV cache across neighboring attention layers, which further reduces KV cache size by a factor of 2-3x」；全域層之間跨 block 共用。
- **int8**：「We use int8 quantization on model weights, activations, and attention KV cache … we natively train our models in int8 precision, eliminating the risk of training/serving mismatch」。
- **快取**：KV 存在 host memory，跨輪次重用，rolling hash 前綴樹＋LRU，sticky session，「Our system achieves a 95% cache rate」。
- 〔判讀〕這是「架構先縮 20 倍，再用 int8」的典型。int8 是**訓練時就用 int8**，不是事後量化，所以不能直接拿來證明「事後量化到 8 bit 沒問題」。品質只有一句「without regressing quality」，**沒有數字**〔業者宣稱〕。

### 3.2 DeepSeek

- **MLA**（DeepSeek-V2，arXiv 2405.04434）〔部分〕：
  - 摘要寫 MLA「reduces the KV cache by 93.3%」（對 DeepSeek 67B）。
  - Table 1：每 token 的 KV 元素數，MHA 是 2·n_h·d_h·l、GQA 是 2·n_g·d_h·l、MQA 是 2·d_h·l、MLA 是 (d_c+d_h^R)·l ≈ (9/2)·d_h·l，「equal to GQA with only 2.25 groups」。
  - §3.2.3 部署：參數轉成 FP8，並做 KV 量化，「compress each element in its KV cache into 6 bits on average」。**沒有給這個 6 bit 的精度數據**。
- **DeepSeek-V3 硬體論文**（ISCA'25，arXiv 2505.09343）〔部分〕：BF16 下每 token KV，DeepSeek-V3（MLA）70.272 KB、Qwen-2.5 72B（GQA）327.680 KB（4.66 倍）、LLaMA-3.1 405B（GQA）516.096 KB（7.28 倍）。
  - 〔算術〕用 HF config 驗算：V3 是 61 層 ×（512＋64）× 2 B＝70,272 B；Qwen2.5-72B 是 80×8×128×2×2＝327,680 B；405B 是 126×8×128×2×2＝516,096 B。**三個都對上**。
  - 這篇把「Quantized Compression」列為業界的 KV 縮減手段之一，但**沒說 DeepSeek 自己的 KV 量化**。
- **線上推論系統**（Open Infra Index Day 6，2025-02-28）〔部分〕：
  - 「matrix multiplications and dispatch transmissions adopt the FP8 format aligned with training, while core MLA computations and combine transmissions use the BF16 format」。
  - 24 小時內輸入 608B token，其中 342B（56.3%）「hit the on-disk KV cache」。
- **KV 存在磁碟**：
  - 2024-08-02 公告「Context Caching on Disk」：存在「distributed disk array」，「made possible by the MLA architecture」，64 token 為單位〔部分〕。
  - 3FS 的 README 把 KVCache 列為用途，「cost-effective alternative to DRAM-based caching」，讀取峰值 40 GiB/s〔部分〕。
- **FlashMLA 的 KV 格式**〔官方文件／程式碼，逐字，本機 README〕：
  - **V3.2**（commit ba89a34 版 README）：656 B／token＝512 個 `float8_e4m3`（NoPE）＋4 個 float32 scale（每 128 個值一個）＋64 個 BF16 的 RoPE，「This part is not quantized for accuracy」。稀疏 decode kernel 讀 FP8，拆回 BF16 再算。
  - **V4**：584 B（448 B FP8＋128 B BF16 RoPE＋e8m0 scale）。
  - **V4.1**（2026-09-10 釋出）：528 B（512 B FP8，RoPE 也量化，每 32 個值一個 e8m0 scale）；FP4 版 288 B（e2m1，每 16 個值一個 e4m3 scale）。「In practice we expect the sliding window (SWA) kv cache to be in FP8 and the compress attention (CA) kv cache to be in FP4」。
  - 現行 README：decode kernel「currently supports only the FP8 and FP4 KV cache formats. Unquantized (bfloat16) KV cache format is not supported」。
- 〔判讀〕DeepSeek 的路線是 MLA（架構縮 KV）→ 6 bit（V2）→ FP8，但保留 RoPE 為 BF16（V3.2）→ FP8 全量化，次要的 KV 放 FP4（V4.1）。每一步都**配合自己寫的 kernel 和訓練**。V3.2 明寫「RoPE 部分為了精度不量化」，這是業者自己承認「不是每個部分都能一樣低」的證據。**DeepSeek 沒有公開任何 KV 量化的答對率數據**（NOT_FOUND）。

### 3.3 Moonshot／Kimi

- **Mooncake**（FAST'25 Best Paper）〔部分〕：「the serving platform for Kimi」，「KVCache-centric Disaggregated Architecture」，「leverages underutilized CPU, DRAM, and SSD resources … to build a disaggregated KVCache pool」；真實負載下「handle 75% more requests」。**KV 精度 NOT_FOUND**。
- **Kimi K2**：HF config 是 `DeepseekV3ForCausalLM`，`kv_lora_rank` 512、`qk_rope_head_dim` 64、61 層〔程式碼／config〕→ 和 V3 一樣是 70,272 B／token〔算術〕。
- **Kimi Linear**（arXiv 2510.26692）〔摘要＋config〕：
  - 摘要寫「layerwise hybrid of KDA and Multi-Head Latent Attention」，「reducing KV cache usage by up to 75%」，1M context 的 decode 吞吐最多 6 倍。
  - config：27 層裡 `kda_layers` 20 層、`full_attn_layers` 7 層（MLA）。〔算術〕1 − 7/27＝74%，和摘要的 75% 對得上。

### 3.4 Google

- **Gemma 3**（arXiv 2503.19786）〔部分〕：「5 local layers for every global layer」，局部層「span of only 1024 tokens」。
  - §5.2：只用全域層時 KV「memory overhead of 60%」，改成 1:3 加 1024 視窗後「less than 15%」（2B、32K）。
  - 改比例對 perplexity「minimal impact」，連 7:1 也是。
  - Table 3 的「+KV」以 32,768 token 計，但**沒寫 KV 的 dtype**。
- **TurboQuant**（Google Research 部落格 2026-03-24；arXiv 2504.19874，ICLR 2026）〔部分〕：
  - 論文摘要：KV「absolute quality neutrality with 3.5 bits per channel」，2.5 bit「marginal quality degradation」。
  - LongBench（Llama-3.1-8B）：16 bit 50.06、TurboQuant 3.5 bit 50.06、2.5 bit 49.44、KIVI 3 bit 48.50、KIVI 5 bit 50.16。
  - NIAH 4K–104K，記憶體 25%：TurboQuant 0.997＝full 0.997，KIVI 0.981。
  - 部落格只說「in models like Gemini」是「a major application」，**沒說 Gemini 有用**〔作者宣稱〕。
  - **獨立結果比較差**：D-Quant 的 RULER 128K，TurboQuant 2.19 bit 在 Qwen3-8B 是 21.18（BF16 72.13）；vLLM 的研究裡 3 bit 推理題掉約 20 分（§5.3）。位元數與題目不同，**只並列**。
- **Gemini 生產的 KV 精度**：NOT_FOUND。

### 3.5 OpenAI

- **gpt-oss**（模型卡，arXiv 2508.10925）〔部分＋config〕：
  - 「attention blocks alternate between banded window and fully dense patterns, where the bandwidth is 128 tokens」；64 個 query head、8 個 KV head、head_dim 64；dense 層用 YaRN 延到 131,072；每個 head 有 learned sink。
  - config：36 層裡 `sliding_attention` 18、`full_attention` 18，`sliding_window` 128。
- 生產環境的 KV 精度：NOT_FOUND。

### 3.6 Meta

- **Llama 4**（2025-04-05 部落格）〔部分〕：「interleaved attention layers without positional embeddings」（iRoPE）、「inference time temperature scaling of attention」、Scout 10M context。**部落格沒提 chunked attention**。
- config（unsloth 的公開鏡像；meta-llama 原倉庫要授權）：48 層，`no_rope_layers` 是 36 個 1、12 個 0，`attention_chunk_size` 8192。transformers 原始碼把 `no_rope_layers=1` 的層設為 `chunked_attention`、0 的設為 `full_attention`（`models/llama4/configuration_llama4.py:198-201`）〔程式碼〕→ 36 層是 8K 分塊的局部注意力、12 層全域。
- 生產環境的 KV 精度：NOT_FOUND（搜尋只找到第三方部落格）。

### 3.7 阿里（Qwen）

- **Qwen3**（技術報告 arXiv 2505.09388 §2）〔部分〕：「we remove QKV-bias used in Qwen2 … and introduce QK-Norm」。
  - 〔判讀〕這和 Mittal 等人（A8）把 Qwen2 系列 FP8 KV 崩潰歸因於「lacking QK-norm」（§5.1）是同一件事。**我們用的 Qwen2.5-7B-1M 屬於沒有 QK-Norm 的那一代**。
- **Qwen3-Next-80B-A3B**（模型卡＋config）〔部分〕：
  - 「12 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))」；Gated Attention 是 16 個 Q head、**2 個 KV head**、head_dim 256；原生 262,144。
  - Base 模型「10 times inference throughput for context over 32K」。
- **Qwen3.5-397B-A17B** config：60 層裡 `linear_attention` 45、`full_attention` 15，2 個 KV head、head_dim 256〔程式碼／config〕。
- 生產環境的 KV 精度：NOT_FOUND。Qwen 團隊有和 SGLang、NVIDIA 合寫 NVFP4 KV 評測（§2.2），但那是實驗性功能的評測，不是生產揭露。

### 3.8 MiniMax

- **MiniMax-01**（arXiv 2501.08313）〔摘要＋config〕：config 的 `attn_type_list` 是 80 層裡 70 層 0（lightning）、10 層 1（softmax）→ 7:1。
- **MiniMax-M2**（「Why Did M2 End Up as a Full Attention Model?」，2025-10-30）〔部分〕：
  - 結論「efficient attention still has some way to go before it can definitively beat full attention」。
  - 「Linear attention is currently far more sensitive to numerical precision than full attention」，列出「Low-Precision State Storage」是待解問題；hybrid 模型「had clear deficits in complex, multi-hop reasoning tasks」。
  - M2 的 config：62 層全部是全注意力、8 個 KV head〔config〕。
- 〔判讀〕這是「架構縮 KV」趨勢的**反例**，而且理由裡直接提到「低精度」。架構縮 KV 和降精度會互相影響，不是可以無限疊加的兩個旋鈕。

---

## 4. 架構層面的 KV 縮減（〔算術〕，用公開 config）

> 算法：BF16（2 B），每層 K＋V。全域層的 KV 隨長度 L 增長；局部／分塊層以視窗 w 封頂（min(L, w)）；線性注意力層是**常數大小**的狀態，這裡不計入。MLA 每層存 (kv_lora_rank＋qk_rope_head_dim) 個元素。config 存在 `/mlsteam/data/tiara/papers/stage3_L6/configs/`；計算腳本在本 session scratchpad 的 `kvcalc.py`。

| 模型（設計） | 每 token KV（128K 平均） | 128K 一條序列 | 對「全部層都是全域」的反事實 | 原文怎麼說（對照） |
|:--|--:|--:|--:|:--|
| **Llama-3.1-8B**（GQA，8 KV heads × 128，32 層）〔我們的模型〕 | 128 KiB | 16.0 GiB | 對 MHA（32 heads）小 **4.0 倍** | — |
| **Qwen2.5-7B-1M**（GQA，4 KV heads × 128，28 層；`use_sliding_window: false`）〔我們的模型〕 | 56 KiB | 7.0 GiB | — | — |
| **DeepSeek-V3／Kimi-K2**（MLA 512＋64，61 層） | 68.6 KiB（70,272 B） | 8.6 GiB | 對同規格 MHA（128 heads，K 192＋V 128）小 **71 倍** | V2 對 67B 少 93.3%；ISCA'25 表 70.272 KB（O2、O3） |
| **Gemma-3-27B**（62 層，每 6 層 1 全域 → 10 全域；16 KV heads × 128；局部 1024） | 83 KiB | 10.4 GiB | 小 **6.0 倍**（32K 時 5.3 倍） | 2B 模型 60% → <15%（O10） |
| **Llama-4-Scout**（48 層：12 全域、36 分塊 8192；8 KV heads × 128） | 57 KiB | 7.1 GiB | 小 **3.4 倍**（32K 時 2.3 倍） | 部落格只寫 iRoPE（O12） |
| **gpt-oss-120b**（36 層：18 全域、18 局部 128；8 KV heads × 64） | 36 KiB | 4.5 GiB | 小 **2.0 倍** | 模型卡（O11） |
| **Qwen3-Next-80B-A3B**（48 層：12 全域、36 GDN；2 KV heads × 256） | 24 KiB＋常數狀態 | 3.0 GiB | 小 **4.0 倍** | 「10 times throughput >32K」（O13） |
| **Qwen3.5-397B-A17B**（60 層：15 全域、45 GDN；2 KV heads × 256） | 30 KiB＋常數狀態 | 3.75 GiB | 小 **4.0 倍** | config（O13） |
| **MiniMax-Text-01**（80 層：10 softmax、70 lightning；8 KV heads × 128） | 40 KiB＋常數狀態 | 5.0 GiB | 小 **8.0 倍** | config（O15） |
| **Kimi-Linear-48B-A3B**（27 層：7 MLA、20 KDA） | 7.9 KiB＋常數狀態 | 0.98 GiB | 對全 MLA 小 **3.9 倍**（−74%） | 「up to 75%」（O9） |
| **MiniMax-M2**（62 層全部全域；8 KV heads × 128）〔反例〕 | 248 KiB | 31.0 GiB | 1 倍 | 回到全注意力（O15） |
| **Character.AI 生產模型** | 沒有 config | — | 合計 >20 倍（MQA 8 倍 × 跨層 2–3 倍 × 局部 1024） | 業者自述（O1） |

**讀法**
- 〔判讀〕2025–26 年的新模型，靠架構就把 KV 縮了 2–71 倍，**都比量化的 2 倍（FP8）或 4 倍（4 bit）大**。量化是疊在上面的第二層（DeepSeek：MLA＋FP8；Character.AI：MQA／局部／共用＋int8）。
- 〔判讀〕我們的兩個模型是「全部層都是全域的 dense GQA」，每 token 的 KV 在上表屬於偏大的一端（Llama 128 KiB 比 DeepSeek-V3 的 68.6 KiB 還大，雖然參數少很多）。所以**「KV 放不下」在我們的模型上比在新架構上嚴重**。這是適用範圍的限制，論文裡要寫清楚。
- 〔推論〕在 hybrid 模型上，搬移與重算的對象除了 KV，還有線性層的常數狀態（MiniMax-M2 也點名「低精度狀態儲存」是難題）。這一層在 Tiara 的動作空間裡沒有對應（`ADVISOR_REPORT_20260930.md` 第 6 點已提過）。

---

## 5. 正式的精度資料

### 5.1 廠商與專案（全部是自測）

| 來源 | 格式 | 模型／題目／長度 | 結果 | 條件與限制 | 性質 |
|:--|:--|:--|:--|:--|:--|
| NVIDIA 技術部落格（2025-12-08，R11） | NVFP4 vs FP8 vs BF16 | Qwen3-480B-A35B：LiveCodeBench、MMLU-PRO、MBPP、RULER 64K | 「accuracy loss of less than 1%」；看圖約 RULER 64K：FP16 95.6、FP8 95.5、NVFP4 94.6 | 只有一個模型；NVFP4 attention 前先轉回 FP8；Llama 3.3 70B MMLU：NVFP4 比 MXFP4 高約 5% | 〔部分〕廠商宣稱，非獨立驗證 |
| LMSYS／SGLang＋Qwen＋NVIDIA（2026-09-16，R9） | NVFP4 vs FP8 | Qwen3.5-397B、Qwen3.8-27B；GSM8K、GPQA、AIME25、SWE-bench Verified | −0.08 到 −1.60 分（對 FP8） | 沒有 BF16 基線；scale 1.0、沒校準；實驗性 | 〔部分〕廠商宣稱，非獨立驗證 |
| vLLM（2026-04-22，R5） | FP8（不校準，scale 1.0） | Llama-3.3-70B、Qwen3-30B-A3B、Qwen3.5-27B、Kimi-K2.5；MRCR 到 1M；128K NIAH | NIAH 91 → 13（FA3 累加問題）→ 修後 89；MRCR 保留 94–98% | H100／B200／H200；沒有 AMD | 〔部分〕專案自測 |
| vLLM（2026-05-11，R6） | FP8、TurboQuant k8v4／4bit／k3v4／3bit | 同上＋MiniMax-M2.7；MRCR 到 256K、AIME25 等 | FP8、k8v4 保留 >98%；4bit-nc 約 96%；3 bit 掉約 20 分（推理題）；MRCR 256K 45.8 → 31.2（3bit-nc） | 只有 H100 | 〔部分〕專案自測 |
| Hugging Face 部落格（Mittal 等，2026-08-06，A8；作者單位文中沒寫） | FP8（e4m3，vLLM `--kv-cache-dtype fp8`） | 10 個模型、RULER 4K–32K | 8 個模型在 4K 的變化 ±0.013 以內；**DeepSeek-R1-Distill-Qwen-7B 0.5291 → 0.0891**；Gemma-3-1B 0.2993 → 0.2082 | Intel Arc Pro B70；scale 設定沒寫；Qwen2.5-7B 沒測（測的是 14B、72B） | 〔部分〕作者自測 |
| AMD ROCm 部落格（2026-08-28，R19） | TurboQuant 4 bit＋BF16 頭尾 | MiniMax-M2.5；RULER 16K–128K | 對 BF16 約 1 分以內 | 只有 MI355X | 〔部分〕廠商宣稱，非獨立驗證 |
| AMD 模型卡（R18） | W8A8＋KV 全 FP8 | Llama-3.1-8B；WikiText-2 | PPL 7.2169 → 7.2752 | 權重和 KV 混在一起；pseudo-quant | 〔部分〕廠商宣稱 |
| Google TurboQuant（O16） | 2.5／3.5 bit | Llama-3.1-8B、Ministral-7B；LongBench、NIAH 到 104K | 3.5 bit＝16 bit；2.5 bit −0.6（LongBench） | 作者自測 | 〔部分〕作者宣稱 |
| llama.cpp PR #7412（R13） | q8_0／q5_0／q4_0 | 沒寫模型、長度；PPL＋KLD | q8_0 KLD 0.00098；q4_0 PPL +3.3%〔算術〕 | 研究用 demo | 〔部分〕 |

### 5.2 MLPerf Inference（MLCommons 規則，master commit `d3eba2f`，2026-10-20 下載）

- **KV 可以量化，不可以 prune**〔官方文件，逐字，`inference_rules.adoc:907-909`〕：「The entries of the KV-cache should be handled in the same way as the activations of a forward pass. They can be quantized according to the quantization rules. However, according to the model equivalence rules, they cannot be pruned (or sparsified). It should be noted that pruning is different from not using a KV-cache (or caching only some entries while rematerializing others); pruning alters the computation and the model's predictions.」
  - 〔判讀〕MLPerf 允許「只快取一部分、其餘重算」（＝我們的 DROP＋重算是合法的），**不允許丟掉不重算**（＝eviction 類方法在 closed division 不合法）。這和我們把 DROP 定義成「丟掉之後要重算」一致。
- **跨 query 不能共用 KV**（`:895-897`）：「it does not apply across queries」→ prefix caching／跨請求重用不在 MLPerf 的範圍。
- **量化規則**（`:680-704`）：closed division 只能用官方校準集，量化方法要公開到「could be reproduced」；open division「any quantization scheme is allowed that achieves the desired quality」。
- **門檻**（`:265-276`）：
  - Llama2-70B「99.9% of FP32」（OpenOrca，`max_seq_len=1024`）
  - Llama3.1-8B「99% … and 99.9% of FP32」（CNN Dailymail，2048）
  - Llama3.1-405B「99% of FP16」（LongBench、RULER、GovReport 子集）
  - DeepSeek-R1「99% of FP16」
  - GPT-OSS-120B「99% of 83.13%」
- **有誰用 FP8 KV**：
  - NVIDIA v4.1 Llama2-70B 的 README：`quantize.py --qformat fp8 --kv_cache_dtype fp8 … --calib_size 1024`〔程式碼，逐字，`inference_results_v4.1/closed/NVIDIA/code/llama2-70b/tensorrt/README.md:33`〕。v5.0、v5.1 的 README 改成「quantization is done in the generate_engines step」，KV dtype 沒寫（NOT_FOUND）。
  - NVIDIA v5.1 DeepSeek-R1：「the key-value (KV) cache is stored in the BF16 data format … the KV-cache was quantized to FP8 precision」〔部分，M4〕。
  - AMD v4.1 MI300X Llama2-70B-99.9：「We also added FP8 support to vLLM, upgraded the hipBLASLt library, and implemented FP8 KV cache」〔逐字，M2，本機存檔〕。
  - Llama3.1-405B（長上下文題）的提交有沒有用 FP8 KV：NOT_FOUND（這輪沒查）。
- 〔判讀〕MLPerf 是**唯一有第三方規則、而且門檻寫死**的正式資料。但能確認用 FP8 KV 的題目，Llama2-70B 輸入 ≤1,024 token、DeepSeek-R1 主要是長輸出，**都證明不了 32K–128K 輸入時 FP8 KV 沒問題**。
- 〔算術〕MLPerf 的門檻是相對的：在我們 Llama 32K 的基線 68.2% 上，99% 等於只能掉 0.68 個百分點、99.9% 是 0.068；我們的 ε（CI 下界 > −2 個百分點）約等於 97.1%。

### 5.3 學術的系統性評測（重點：位元數 vs 長上下文答對率，有沒有懸崖）

| 論文 | 模型 | 長度 | 位元／格式 | 主要結果 | 懸崖在哪 | 深度 |
|:--|:--|:--|:--|:--|:--|:--|
| Li et al.，「Evaluating Quantized LLMs」（arXiv 2402.18158；論文 metadata 標 ICML，會議頁未另核） | 11 個模型家族 | 長文本＝≥4K（LongEval 等） | KV8／4／3／2 | 「The longer the text, the larger the performance loss caused by Weight and KV Cache quantization」；多數模型「more sensitive to KV Cache Quantization」；建議 <4K 用 KV4、**≥4K 用 KV8**；LongChat 連 KV8 都受傷 | KV2 多數大掉；長文本 KV4 就開始掉 | 〔部分〕 |
| Yuan et al.，「KV Cache Compression, But What Must We Give in Return?」（arXiv 2407.01527；本專案 05_QA 記 Findings of EMNLP 2024） | Llama-3-8B-Inst、Mistral-7B、LongChat-7B | LongBench（最長 31.5K）、needle 約 27–31K token | KIVI-2、KIVI-4、FlexGen-4 | Llama LongBench：45.2 → KIVI-4 45.3、KIVI-2 44.3；needle 都 100；「Quantization methods can often achieve reliable performance across all task categories」 | 2 bit 只小掉（≤32K） | 〔部分〕 |
| Zhang et al.，「More Tokens, Lower Precision」（arXiv 2412.12706） | Llama-3-8B、Mistral-7B（加 1B–70B 的規模實驗） | RULER **只有 4K／8K／16K**、LongBench、NIAH | 16／8／4／2 bit × 各種剪枝 | 「reducing precision to 2-bit leads to a drastic performance decline」；4 bit「minimal」、8 bit「negligible」；同記憶體下「4 bit、4 倍 token」最好 | **2 bit** | 〔部分〕 |
| Hooper et al.，KVQuant（NeurIPS 2024，arXiv 2401.18079 v6） | LLaMA-2-7B-32K | **RULER 32K**、LongBench 31.5K | 3 bit／2 bit（各加 1% 離群值）；KIVI-2 | RULER：fp16 56.40；KVQuant-3bit 53.65（−2.75）；KIVI-2（3.05 bit）39.78（−16.62）；KVQuant-2bit 36.54（−19.86）。LongBench：31.96 → 3 bit 31.21、KIVI-2 30.04 | **3 → 2 bit 之間**（RULER 比 LongBench 敏感） | 〔部分〕 |
| Su et al.，D-Quant（arXiv 2609.19880，2026-09 預印本） | Llama-3.1-8B、Qwen3-8B | **RULER 32K／64K／128K** | 只有約 2 bit 的基線（KIVI 2.25、QuaRot 2.25、TurboQuant 2.19） | Llama BF16 89.51／85.96／76.44 vs KIVI 75.37／69.18／51.87（−14.1／−16.8／−24.6）；Qwen3 BF16 91.29／78.78／72.13 vs KIVI 69.43／48.33／33.83（−21.9／−30.5／−38.3）；「accuracy deteriorates rapidly as the sequence length increases」 | 2 bit，**而且越長越糟** | 〔部分〕預印本 |
| Xia et al.，Kitty（arXiv 2511.18643，2025-11 預印本） | Qwen3、LLaMA3 兩個家族 | 7 個任務（長度沒在摘要） | 2 bit vs 4 bit | 「While 4-bit KV quantization preserves accuracy, 2-bit often degrades it, especially on long-context reasoning」 | 2 bit | 〔摘要〕 |
| Salammagari，「Length, Not Bit Width」（NeurIPS 2026 長上下文 workshop poster，單一作者） | Llama-3.1-70B、Qwen-2.5-72B | 4K–128K 單針撈取，190 次 | FP16、int8、int4（affine） | 量化把撈取的懸崖「往內移」：Qwen FP16 撐到 96K，量化後只到 32K；int8 和 int4 的通過／失敗型態幾乎一樣；該實作的峰值記憶體反而更高、128K prefill 吞吐約減半 | **由長度決定**，不是位元數 | 〔摘要〕workshop，未經主會議審查 |
| Turganbay，HF 部落格（2024-05-16，A9） | Llama2-7b-chat | LongBench（模型 4K） | quanto int4、int2 | int4 ≈ fp16；int2：SAMSum 41.12 → 14.04、TriviaQA 84.28 → 63.64 | 2 bit | 〔部分〕官方部落格 |
| vLLM TurboQuant 研究（R6） | Qwen3-30B-A3B、Llama-3.3-70B、MiniMax-M2.7 | MRCR 到 256K | 8／4／3 bit | 3 bit：推理題掉 8–20 分（依模型）、MRCR 256K 掉 12.3／14.6 分；4 bit（4bit-nc）：MRCR −3.5、推理保留約 96% | **3 bit** | 〔部分〕 |
| Liu et al.，ShotKV（arXiv 2502.01941，ICML 2026） | — | 長上下文生成 | 各種「aggressive compression」 | 檢索類「remain robust」，推理類「severe Task-Dependent Degradation」 | 依任務 | 〔摘要〕 |
| Gao et al.，「Rethinking KV Cache Compression…」（MLSys 2025） | — | — | — | 壓縮可能讓輸出變長、端到端延遲反增；現有實作不是為生產級服務最佳化 | — | 〔摘要〕 |

**讀法**〔判讀〕
1. **有「懸崖」，位置大約在 2–3 bit**（K 逐 channel 分組的好格式，如 KIVI、KVQuant）；4 bit 以上多半在雜訊內。≥32K 的資料（KVQuant 32K、D-Quant 32–128K、vLLM 256K）都支持這一點，而且 **D-Quant 和 workshop 的資料顯示懸崖會隨長度變嚴重**。
2. **但 Li et al.（≥4K 建議 KV8）和 workshop（int8 也讓懸崖內移）說「8 bit 也不是一定安全」**，取決於模型。這和下面我們自己的 Qwen 結果、Mittal 等人（A8）的 Qwen2 系列結果一致：**有 K 離群 channel 的模型，粗分組的 8 bit 也可能崩**。
3. **≥32K、同一個模型、同一種題目、把 8／4／3／2 bit 整條掃過的獨立研究：NOT_FOUND**。最接近的是 vLLM 的 TurboQuant 研究（8／4／3 bit，到 256K，但只在 H100、作者是 vLLM 團隊）和 KVQuant（32K，只有 3／2 bit）。我們第二階段的長 context 掃描（§5.4，13 種格式 × 32K／64K／128K × 兩個模型）在這一點上反而比公開文獻完整，但每格只有 48 題。

### 5.4 本專案自己的資料（MI300X，第二階段；不重量，只轉錄）

| 格式（位元／值） | Llama 32K／64K／128K（對 E0，百分點） | Qwen 32K／64K／128K | 出處 |
|:--|:--|:--|:--|
| E0（BF16）答對率 | 68.2／66.7／40.3% | 77.4／61.9／56.6% | P1 §1 |
| FP8 逐 token（Llama）／**逐 head（Qwen）**，8.25／8.00 | +1.2／−0.9／+1.6 | **−37.9／−44.9／−55.1** | P1 §2 |
| FP8 扣 K 平均（fp8k0） | −0.3／−0.6／+0.7 | −1.0／−0.4／−0.6 | 同上 |
| INT8 逐 token，8.25 | 0.0／−0.4／+1.9 | −0.4／−0.7／−2.5 | 同上 |
| **INT4 KIVI**，5.00 | +1.0／+0.2／+1.3 | −0.9／−0.8／−0.1 | 同上 |
| INT4 逐 token、MXFP4、NVFP4，4.25–4.50 | 在 −8.1 到 +4.5 之間；NVFP4 64K −6.6 | **全部＝−E0（答對率 0%）** | 同上 |
| **INT2 KIVI**，3.00 | +0.4／−4.2／**−7.2 [−13.5, −1.4]** | **−12.9 [−22.1, −4.9]**／−0.1／−12.5 | 同上 |

- run：Llama `20261019-033004-m10-lc-screen-llama-252-s0` 等 6 個；Qwen `20261019-055235-m10-lc-screen-qwen-252-s0` 等 6 個（`results/m10_longctx_mi300x/analysis/tables_longctx.md` 開頭）〔實測 run〕。每個長度 48 題，**探索，不判定**。
- 32K 主線（n＝180×3 族＋T5 60）：Qwen FP8 逐 head −32.8 [−35.8, −30.0]；同模型 INT8 +0.2、FP8 扣 K 平均 +0.4、INT4 KIVI +0.1（RUNLOG_MI300X 10.8、10.10）〔實測 run〕。
- **vLLM 原生 FP8 在 MI300X 上的品質**（九月，舊設計，有 05_QA §3 列的設計問題）：
  - Llama 64K／129K 撈針五種精度都 20/20。
  - Qwen 約 128K 撈針：BF16 20/20、vLLM `fp8` 2/20、`fp8_per_token_head` 0/20、`int8_per_token_head` 18/20、`int4_per_token_head` 0/20（run `20260916-065920-m5-b-qwen7b-needle`；原始 CSV 已遺失，由 F4 稽核轉錄，`01_advisor_feedback.md` §3）〔實測 run〕。
- 〔判讀〕我們的資料和 §5.3 的懸崖一致（INT2 開始掉），但多了一個公開文獻很少明講的點：**在 Qwen2.5 上，格式的分組方式比位元數重要**。8 bit 的逐 head FP8 會壞，4 bit 的 K 逐 channel KIVI 不會。這正好也是業界「預設 FP8」最容易出事的模型家族（§5.1 的 Intel 結果）。

---

## 6. 回答「是不是越低越好」

### 6.1 好處：位元越低，記憶體與搬運越省（〔算術〕，我們的模型）

| 格式（位元／值） | Llama 128K 一條序列 | Qwen 128K 一條序列 | 對 BF16 |
|:--|--:|--:|--:|
| BF16（16） | 16.0 GiB | 7.0 GiB | 1× |
| FP8／INT8（約 8–8.25） | 約 8.0–8.3 GiB | 約 3.5–3.6 GiB | 約 1/2 |
| INT4 KIVI（5.00，含 scale） | 約 5.0 GiB | 約 2.2 GiB | 約 1/3.2 |
| INT2 KIVI（3.00） | 約 3.0 GiB | 約 1.3 GiB | 約 1/5.3 |

位元組少多少，搬到 CPU／SSD 的時間也大致少多少——**前提是搬運是瓶頸**。第二階段量過：限速層裡 INT4 比 BF16 快 36–57%；但在真實 CPU 層，原寫法的反量化讓 INT4 反而更慢，換成融合 kernel 才翻回來（RUNLOG 10.9、08 §0 第 1 列）〔實測 run〕。

### 6.2 代價：品質懸崖＋反量化成本

- **品質**：§5.3、§5.4。好的 4 bit 以上多半無損；2–3 bit 開始掉，而且越長越糟；有離群值的模型連粗分組的 8 bit 都可能壞。
- **成本**：
  - vLLM／H100：FP8 吞吐＝BF16，TurboQuant 剩 66–80%，延遲 +10–68%（R6）〔部分〕。
  - vLLM：`head_dim=256` 的模型開 FP8 後長 context 的 TTFT 係數約 1.6 倍（R5）〔部分〕。
  - workshop：該實作量化後峰值記憶體更高、128K prefill 吞吐約減半（A7）〔摘要〕。
  - 我們的 MI300X：vLLM 原生 FP8 decode 每 token 快 4–29%、prefill＋1 慢 2–9%；我們自寫的 Triton「拆進 attention」讀 INT4，Llama 32K 每 token 7.11 vs 5.26 ms，2–5 個 token 就把還原省下的時間賠掉；GPU 忙時 FP8 反量化慢 2.4–3.8 倍（L2 §0.1、`docs/precision_20261013/09_sweep_plan.md` §6.1）〔實測 run〕。
- **一句話**〔判讀〕：位元降一半，位元組少一半，但品質不是線性的，有一個依模型而定的懸崖；kernel 成本則要看硬體有沒有原生格式。**「越低越好」只在「品質還沒到懸崖、而且反量化不搶關鍵路徑」的區間成立**。

### 6.3 主流停在哪裡、為什麼

- **停在 FP8**（§0 第 1 點的六個來源）。理由〔判讀，各點有出處〕：
  1. **硬體原生**：FP8 從 Ampere（只儲存）到 Hopper、Ada、Blackwell、MI300 都有路徑（R10 硬體矩陣、R16）。NVFP4 只有 Blackwell SM100／103（R10）；vLLM 的 `nvfp4` 只在 FlashInfer，ROCm 用不了（R3）。
  2. **kernel 不變慢**：vLLM 的 FP8 吞吐等於 BF16（R6），我們在 MI300X 上 decode 反而快（RUNLOG 14.4(d)）。
  3. **一般模型幾乎無損**：vLLM 自測、Mittal 等人 10 個模型中 8 個、SGLang 文件（R5、A8、R7）。
  4. **2 倍剛好夠用**：架構已經縮了 2–71 倍（§4），再省一半通常就夠。
- **為什麼不再往下（目前）**：
  - 4 bit 要嘛需要原生格式（NVFP4 在 Blackwell），要嘛付反量化成本（TurboQuant 吞吐 66–80%）。
  - 3 bit 以下有品質懸崖（§5.3）。
  - 連 DeepSeek-V4.1 也只把次要的「壓縮注意力」KV 放 FP4，滑動視窗那部分仍是 FP8（O7）。
- **為什麼也不是人人都開 FP8**：
  - 所有引擎預設都還是 16 bit。
  - vLLM 自己建議 <7K context、`head_dim=256` 在乎 prefill、不校準時掉到 95% 以下，就留在 BF16（R5）；TurboQuant 研究也說記憶體不是瓶頸時用 BF16（R6）。
  - 有已知會崩的模型（Qwen2 系列、Gemma-3-1B，A8）和 kernel 問題（FA3 累加，R5）。

### 6.4 哪些是正式的、哪些只是宣稱

| 說法 | 證據 | 等級 |
|:--|:--|:--|
| MLPerf 允許 KV 量化、不允許 prune；門檻 99%／99.9% | 規則原文（M1） | **正式規則** |
| FP8 KV 能過 MLPerf 門檻 | NVIDIA、AMD 的提交（M2–M4） | **正式**，但只有短輸入（Llama2-70B ≤1,024）與 DeepSeek-R1 |
| 長文本對 KV 量化更敏感；2 bit 有懸崖 | Li et al.、KVQuant、Zhang et al.（A1、A4、A3）；D-Quant、Kitty（預印本） | **同儕審查＋預印本**，多個獨立團隊 |
| FP8 在一般 GQA 模型上近乎無損 | vLLM（R5）、Mittal 等人（A8）、本專案 Llama（§5.4） | 多來源，**大多是自測**，沒有同儕審查 |
| Qwen2 系列在 FP8 KV 上會崩 | Mittal 等人（A8）、本專案（§5.4）、KVTuner／DiffKV 對 4 bit 的報告（本專案 05_QA §4） | 兩個獨立來源一致＋論文間接支持 |
| NVFP4 KV 損失 <1% | NVIDIA（R11）、LMSYS（R9） | **廠商宣稱，非獨立驗證**；單一模型或只對 FP8 比 |
| TurboQuant 3–3.5 bit 無損 | Google（O16） | **作者宣稱**；獨立結果（D-Quant、vLLM R6）在 2–3 bit 看到明顯下滑 |
| AMD：FP8 KV「minimal accuracy impact」 | ROCm 文件（R16） | **廠商宣稱**，沒有附數據 |
| Character.AI int8「without regressing quality」 | 業者部落格（O1） | **業者宣稱**，沒有數字；而且是原生 int8 訓練 |
| 多少比例的部署開 FP8 KV | — | **NOT_FOUND** |

---

## 7. 對我們實驗設計的意義

### 7.1 「實際部署預設」對照組：BF16 還是 FP8？

- **主對照組＝BF16（vLLM `auto`）**。理由：
  1. 它是 vLLM、SGLang、llama.cpp 的**預設**（§2），也是兩個模型的原生 dtype。
  2. 第二階段的 E0、I3 的「可部署基線」（vLLM 原生分層卸載）都是 BF16，換掉會讓前後不可比。
  3. MLPerf 也是以 FP16／FP32 參考實作為 100%。
- **建議加一個副對照組＝「常見部署」：vLLM 原生 FP8 KV**。理由：
  1. §0 第 1 點的六個來源都把 FP8 當實際部署的起點，審稿人一定會問「直接開 FP8、容量翻倍，還需要你的分層嗎？」。
  2. 〔推論〕FP8 讓每 token 的位元組減半，κ（重算÷傳輸）會往「傳輸」那邊移約 2 倍，**可能翻轉放置決定**。所以 κ 至少要用 BF16 和 FP8 兩種位元組各算一次。這可以只用算術做，第二階段已經有 FP8 的還原計時（RUNLOG 10.2–10.6）。
- **但 Qwen2.5-7B-1M 不能用 vLLM 的 `fp8` 當「部署預設」**：
  1. vLLM 在 MI300X 上不校準時是 per-tensor、scale＝1.0、e4m3fnuz（最大 240）〔程式碼〕；Qwen 的 max|K| 約 422（RUNLOG:321）。
  2. 九月實測：vLLM `fp8` 撈針 2/20，`int8_per_token_head` 18/20（§5.4）。
  3. 第二階段我們自己的逐 head FP8 在 32K–128K 掉 33–55 個百分點；Mittal 等人在 Qwen2 系列也看到同樣的崩潰（A8）。
  4. → Qwen 的副對照組建議用 **`int8_per_token_head`**（vLLM 0.28 在 MI300X 上走 TRITON_ATTN，容量 1.939 倍，F3 表）。或者照實寫「FP8 在此模型失效，以 INT8 代替」，**不要拿已知會壞的設定當稻草人**。
- **Llama 的 FP8**：
  - 我們自己的 FP8（逐 token 動態 scale）在 32K–128K 都在 ±2 以內。
  - **vLLM 原生 FP8（per-tensor、scale＝1.0）在我們的題目上的答對率：NOT_MEASURED**；只有九月 64K／129K 撈針 20/20，這是很簡單的題。
  - 兩者的 scale 方式不同，**不能互相代替**〔判讀〕。

### 7.2 MI300X＋vLLM 0.28 上 FP8 KV 能不能用（只用 repo 已有紀錄）

| 問題 | 答案 | 出處 |
|:--|:--|:--|
| `environment_snapshot.md` 有沒有記 FP8 KV | **NOT_FOUND**：只有 ROCm、vLLM 版本，以及 connector 退回 TRITON_ATTN | `docs/stage3_20261019/environment_snapshot.md` §3 |
| 開得起來嗎 | 可以。log「Using fp8 data type to store kv cache」；走 ROCM_ATTN | `docs/research_20261010_followup/F3_dequant_fp8.md` §4.6〔實測 run `20261010-074830-f3-fp8`，九月 `20260916-062139`、`20260917-103133` 的 server.log〕 |
| 容量 | Llama 1,267,504 → 2,535,008 token（2.000 倍，154.73 GiB）；`fp8_per_token_head`、`int8_per_token_head` 2,458,192（1.939 倍）；`int4_per_token_head` 4,771,792（3.765 倍） | 同上 |
| FP8 有沒有真的生效 | 有：4/4 個 prompt 的 logprob 和 BF16 不同，BF16 重跑差 0 | F3〔實測 run `20261010-074830-f3-fp8`〕 |
| decode／prefill 速度 | Llama：32K 5.27 vs 5.88 ms／token（−10%）、128K 6.91 vs 9.76（−29%），prefill＋1 慢 2–6%。Qwen（noDCA）：32K 4.34 vs 4.53（−4%）、128K 4.97 vs 5.75（−14%），prefill＋1 慢 3–9% | `results/RUNLOG_MI300X.md` 14.4(d)〔實測 run `20261018-153119-…-q5d-llama-auto`、`-153807-…-llama-fp8`、`20261018-162351-…-q5d-qwen-auto-nodca`、`-162925-…-qwen-fp8-nodca`〕 |
| Qwen 能不能跑 | 要用 noDCA 變體；原模型兩個 dtype 都會 `TypeError … 'layer_idx'` | 同上 |
| 接 OffloadingConnector 時 | vLLM 會停用 ROCM_ATTN（`platforms/rocm.py:482-485`），改走 TRITON_ATTN（33K attention 1.367 → 3.277 s，RUNLOG:176）；**FP8＋connector 的速度與品質：NOT_MEASURED**；OffloadingConnector 搬 FP8 page 有沒有問題：NOT_VERIFIED | 程式碼＋RUNLOG |
| 照 AMD 文件下指令 | 文件的 `--calculate-kv-scales` 在 0.28 已移除 → 不能照抄 | §2.1、§2.6 |
| NVFP4／MXFP4 KV | vLLM 0.28 在 ROCm 上沒有（只有 FlashInfer）；MI300X 也沒有原生 FP4 | §2.1 |

### 7.3 具體建議（照成本排序；全部不需要新的 GPU 時間，除了最後一條）

1. **論文措辭**：寫「以 16 位元 KV（vLLM 預設）為主基線；FP8 KV 是業界常見的部署選項（引 R5、R6、R16、M2–M4），我們另報 FP8 位元組下的 κ 與決策」。**不要寫「業界都用 FP8」**——沒有部署比例的資料（NOT_FOUND）。
2. **κ 雙算**（只要 CPU，〔算術〕）：用第二階段已有的 FP8 還原計時與重算計時，算 BF16 與 FP8 兩種位元組下的 κ，看放置決定翻不翻。
3. **適用範圍一段**：我們的模型是「全層全域的 dense GQA」，Llama 每 token 的 KV（128 KiB）是 §4 表中 2025–26 新架構（MiniMax-M2 除外）的 1.5–16 倍，Qwen（56 KiB）在中間；在 MLA、局部注意力、線性混合的模型上，KV 本來就小，「放不下」的壓力會低很多，而且線性層的狀態不在 Tiara 的動作空間裡。
4. **把 Qwen 的 FP8 失效寫成發現**：它和 Mittal 等人（A8）的 Qwen2 系列結果、Qwen3 加 QK-Norm 的設計改動對得上。這支持「放置／精度決策要依模型而定」，**不能把 FP8 當通用安全預設**。這是 Tiara「品質約束下的決策」的一個好例子。
5. **（有 GPU 空檔才做，第二階段佇列優先）**：Llama 用 vLLM 原生 `--kv-cache-dtype fp8`、Qwen 用 `int8_per_token_head`，在第二階段同一批 T2–T5 題目上各跑一次 32K 品質。這樣副對照組就有「vLLM 原生」的答對率，不必借我們自己 codec 的數字。估計：每個模型的 32K 品質約數小時（照 RUNLOG 10.8 的節奏推）〔推論〕。

### 7.4 審稿人可能會問的

- 「為什麼不直接開 FP8／NVFP4 就好？」→ 7.1、7.3 第 2 點：位元組減半以後 κ 仍然跨硬體變動；NVFP4 不在 AMD 上；Qwen2 系列的 FP8 會崩。
- 「為什麼 DROP 是合法動作？」→ MLPerf 規則把「只快取一部分、其餘重算」和 prune 分開，前者合法（§5.2）。
- 「MLPerf 門檻 99% 你們的 ε 是多少？」→ 我們的 ε 是絕對 −2 個百分點，在 68% 基線上約等於 97.1%，比 MLPerf 寬。要嘛說明理由（題數與 CI），要嘛另報 99% 門檻下的結果。

---

## 8. 參考清單

> 「讀法」欄：〔全文〕〔部分〕〔摘要〕的定義見檔頭。所有網址都在 2026-10-20 用 WebFetch 或 curl 實際開過；本機存檔見 §9。

**引擎、文件、程式碼**

| # | 出處 | 讀法 |
|:--|:--|:--|
| R1 | vLLM v0.28.0 原始碼（commit 2cf0a69，2026-08-24；本機 `/mlsteam/workspace/src/vllm-v0.28.0`）：`vllm/config/cache.py:19-37, 77-86, 271-289`；`vllm/utils/torch_utils.py:77-87` | 〔程式碼〕 |
| R2 | vLLM 文件「Quantized KV Cache」：本機 `docs/features/quantization/quantized_kvcache.md`（0.28.0）；線上 <https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/>（頁尾 2026-08-03） | 〔官方文件〕〔全文（本機）／部分（線上）〕 |
| R3 | vLLM v0.28.0：`v1/attention/backends/{rocm_attn.py:171, rocm_aiter_fa.py:720, rocm_aiter_unified_attn.py:31, triton_attn.py:296-306, 536-557, flashinfer.py:408-417, flash_attn.py:79, turboquant_attn.py:133}`；`platforms/rocm.py:481-495, 976-983`；`model_executor/layers/quantization/kv_cache.py:95-151`；`models/deepseek_v32/attention.py:162, 273-285`；`models/deepseek_v4/attention.py:104-114`；`model_executor/layers/attention/mla_attention.py:349-361` | 〔程式碼〕 |
| R4 | vLLM PR #49389（移除 `--calculate-kv-scales`，2026-08-03 合併）——本專案 `docs/precision_20261013/cards/P04_vllm_fp8_calibration.md` §1.2 已核 | 〔官方文件〕（轉引） |
| R5 | J. Kübler, E. Kurtić, L. Wilkinson, M. Bonanni, M. Goin, A. Marques, K. Budhathoki，「The State of FP8 KV-Cache and Attention Quantization in vLLM」，vLLM Blog，2026-04-22，<https://vllm.ai/blog/2026-04-22-fp8-kvcache> | 〔部分〕 |
| R6 | E. Kurtić, M. Goin, A. Marques，「A First Comprehensive Study of TurboQuant: Accuracy and Performance」，vLLM Blog，2026-05-11，<https://vllm.ai/blog/turboquant> | 〔部分〕 |
| R7 | SGLang 文件「Quantized KV Cache」，<https://docs.sglang.io/advanced_features/quantized_kv_cache.html> | 〔官方文件〕〔部分〕 |
| R8 | SGLang 原始碼（main，2026-10-20 下載）：`python/sglang/srt/arg_groups/fields/model.py:197-221`；`arg_groups/overrides.py:309-321, 490-535`；`server_args.py:578-597` | 〔程式碼〕 |
| R9 | SGLang、Qwen、NVIDIA 團隊，「Accelerating Long-Context and Agentic Inference with NVFP4 KV Cache」，LMSYS Blog，2026-09-16，<https://www.lmsys.org/blog/2026-09-16-nvfp4-kv-cache> | 〔部分〕 |
| R10 | TensorRT-LLM（main，2026-10-20 下載）：`docs/source/features/quantization.md`、`docs/source/features/kv-cache-compression.md`、`tensorrt_llm/quantization/mode.py:55`；HTML 版 <https://nvidia.github.io/TensorRT-LLM/1.3.0rc13/features/quantization.html> | 〔官方文件〕〔程式碼〕〔部分〕 |
| R11 | E. Alvarez, W.-M. Chen, H. Mao，「Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache」，NVIDIA Technical Blog，2025-12-08，<https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/> | 〔部分〕 |
| R12 | llama.cpp（master，2026-10-20 下載）：`tools/server/README.md`（<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>）；`src/llama-context.cpp:558-559, 4003, 4063-4069` | 〔官方文件〕〔程式碼〕 |
| R13 | J. Gäßler，llama.cpp PR #7412（2024-05-20），<https://github.com/ggml-org/llama.cpp/pull/7412> | 〔部分〕 |
| R14 | LMCache 文件「L2 Serde」，<https://docs.lmcache.ai/mp/serde.html> | 〔官方文件〕〔部分〕 |
| R15 | NVIDIA Dynamo「KVBM design」v1.3.0，<https://docs.nvidia.com/dynamo/v1.3.0/design-docs/component-design/kvbm-design.md> | 〔官方文件〕〔部分〕 |
| R16 | AMD ROCm 文件「vLLM V1 performance optimization」（latest，2026-10-20 下載），<https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html> | 〔官方文件〕〔部分，逐字〕 |
| R17 | C. Greene，「Enhancing vLLM Inference on AMD GPUs」，ROCm Blogs，2024-09-19（2025-06-09 更新），<https://rocm.blogs.amd.com/artificial-intelligence/vllm-optimize/README.html> | 〔部分〕 |
| R18 | AMD，`amd/Llama-3.1-8B-Instruct-FP8-KV` 模型卡，<https://huggingface.co/amd/Llama-3.1-8B-Instruct-FP8-KV> | 〔部分〕 |
| R19 | A. G. Rana 等，「4-bit KV Caching in LMCache: Offloading Quantized KV Beyond HBM for Context-Heavy Agents on AMD MI355X」，ROCm Blogs，2026-08-28，<https://rocm.blogs.amd.com/software-tools-optimization/4bit-KV-LMcache/README.html> | 〔部分〕 |
| R20 | ROCm Composable Kernel `float8.hpp`（FNUZ e4m3 最大 240、OCP e4m3 最大 448），<https://rocm.docs.amd.com/projects/composable_kernel/en/latest/doxygen/html/float8_8hpp_source.html> | 〔摘要〕（搜尋結果摘要，頁面本身沒開） |

**MLPerf**

| # | 出處 | 讀法 |
|:--|:--|:--|
| M1 | MLCommons，「MLPerf Inference Rules」，`inference_rules.adoc`（master commit d3eba2f，2026-10-20 下載），<https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc>：`:265-276`（門檻）、`:680-704`（量化）、`:895-913`（KV-cache Q&A） | 〔官方文件〕〔部分，逐字〕 |
| M2 | M. Arunachalam, M. Hodak，「Engineering Insights: Unveiling MLPerf Results on AMD Instinct MI300X Accelerators」，AMD Blog，2024-08-28，<https://www.amd.com/en/blogs/2024/engineering-insights-unveiling-mlperf-results-on.html> | 〔部分，逐字（本機存檔）〕 |
| M3 | mlcommons/inference_results_v4.1，`closed/NVIDIA/code/llama2-70b/tensorrt/README.md:33`，<https://github.com/mlcommons/inference_results_v4.1> | 〔程式碼〕〔逐字〕 |
| M4 | NVIDIA，「NVIDIA Blackwell Ultra Sets New Inference Records in MLPerf Debut」，2025-09-09，<https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut> | 〔部分〕 |
| M5 | AMD，「Benchmarking Machine Learning using ROCm and AMD GPUs: Reproducing Our MLPerf Inference Submission」，ROCm Blogs，2024-08-28，<https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inf-4-1/README.html>（只寫「KV cache scales … are optimized」，沒明寫 dtype） | 〔部分〕 |

**業者揭露與模型**

| # | 出處 | 讀法 |
|:--|:--|:--|
| O1 | Character.AI，「Optimizing AI Inference at Character.AI」，2024-06-20，原址 `https://research.character.ai/optimizing-inference/`（現轉址；讀 Wayback 存檔）；公告版 <https://blog.character.ai/optimizing-ai-inference-at-character-ai/> | 〔全文〕 |
| O2 | DeepSeek-AI，「DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model」，arXiv 2405.04434（摘要、§2.1 Table 1、§3.2.3） | 〔部分〕 |
| O3 | DeepSeek-AI，「Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures」，ISCA 2025，arXiv 2505.09343 | 〔部分〕 |
| O4 | DeepSeek，「Day 6: One More Thing — DeepSeek-V3/R1 Inference System Overview」，2025-02-28，<https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md> | 〔部分〕 |
| O5 | DeepSeek API News，「DeepSeek API introduces Context Caching on Disk」，2024-08-02，<https://api-docs.deepseek.com/news/news0802> | 〔部分〕 |
| O6 | deepseek-ai/3FS README，<https://github.com/deepseek-ai/3FS> | 〔部分〕 |
| O7 | deepseek-ai/FlashMLA README（main 與 commit ba89a34，2026-10-20 下載），<https://github.com/deepseek-ai/FlashMLA> | 〔官方文件〕〔部分，逐字〕 |
| O8 | Mooncake 專案頁，<https://kvcache-ai.github.io/Mooncake/>（FAST 2025 Best Paper） | 〔部分〕 |
| O9 | Kimi Team，「Kimi Linear: An Expressive, Efficient Attention Architecture」，arXiv 2510.26692；HF config `moonshotai/Kimi-Linear-48B-A3B-Instruct`、`moonshotai/Kimi-K2-Instruct` | 〔摘要〕〔程式碼／config〕 |
| O10 | Gemma Team，「Gemma 3 Technical Report」，arXiv 2503.19786（§1、§2、§5.2、Table 3）；config 取自 `unsloth/gemma-3-27b-it`（google 原倉庫要授權） | 〔部分〕〔config〕 |
| O11 | OpenAI，「gpt-oss-120b & gpt-oss-20b Model Card」，arXiv 2508.10925（§2.2）；HF config `openai/gpt-oss-120b` | 〔部分〕〔config〕 |
| O12 | Meta，「The Llama 4 herd」，2025-04-05，<https://ai.meta.com/blog/llama-4-multimodal-intelligence/>；config 取自 `unsloth/Llama-4-Scout-17B-16E-Instruct`；transformers `models/llama4/configuration_llama4.py:176-201`（本機 venv） | 〔部分〕〔config〕〔程式碼〕 |
| O13 | Qwen，`Qwen/Qwen3-Next-80B-A3B-Instruct` 模型卡與 config，<https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct>；`Qwen/Qwen3.5-397B-A17B` config | 〔部分〕〔config〕 |
| O14 | Qwen Team，「Qwen3 Technical Report」，arXiv 2505.09388（§2） | 〔部分〕 |
| O15 | MiniMax，「MiniMax-01: Scaling Foundation Models with Lightning Attention」，arXiv 2501.08313；「Why Did M2 End Up as a Full Attention Model?」，2025-10-30，<https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model>；HF config `MiniMaxAI/MiniMax-Text-01`、`MiniMaxAI/MiniMax-M2` | 〔摘要〕〔部分〕〔config〕 |
| O16 | A. Zandieh, M. Daliri, M. Hadian, V. Mirrokni，「TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate」，arXiv 2504.19874（ICLR 2026）；A. Zandieh, V. Mirrokni，「TurboQuant: Redefining AI Efficiency with Extreme Compression」，Google Research Blog，2026-03-24，<https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/> | 〔部分〕 |
| O17 | NVIDIA，「NVIDIA Blackwell Delivers World-Record DeepSeek-R1 Inference Performance」，2025-03-18（圖 3 註腳：H200、B200 都用 FP8 KV cache） | 〔部分〕 |

**學術與第三方評測**

| # | 出處 | 讀法 |
|:--|:--|:--|
| A1 | S. Li, X. Ning, L. Wang 等，「Evaluating Quantized Large Language Models」，arXiv 2402.18158（v2；metadata 標 ICML） | 〔部分〕 |
| A2 | J. Yuan, H. Liu, S. Zhong 等，「KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches」，arXiv 2407.01527 v2 | 〔部分〕 |
| A3 | J. Zhang, D. Zhu, Y. Song 等，「More Tokens, Lower Precision: Towards the Optimal Token-Precision Trade-off in KV Cache Compression」，arXiv 2412.12706 v2 | 〔部分〕 |
| A4 | C. Hooper 等，「KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization」，NeurIPS 2024，arXiv 2401.18079 v6（Table 3、4） | 〔部分〕 |
| A5 | Y. Su, H. Liu, G. Yu, J. Zhu，「D-Quant: Driftable Entropy Coding for KV Cache Quantization」，arXiv 2609.19880（2026-09-17，Table 3） | 〔部分〕 |
| A6 | H. Xia, X. Wu, J. Li 等，「Kitty: Accurate and Efficient 2-bit KV Cache Quantization with Dynamic Channel-wise Precision Boost」，arXiv 2511.18643 | 〔摘要〕 |
| A7 | A. R. R. Salammagari，「Length, Not Bit Width: KV-Cache Quantization Moves the Long-Context Retrieval Cliff on a 512 GB Workstation」，NeurIPS 2026 Workshop on Long-Context Foundation Models，<https://neurips.cc/virtual/2026/173781>（沒找到 arXiv 版） | 〔摘要〕 |
| A8 | K. Mittal, C. Tao, L. Uppuluri，「FP8 KV-Cache on Intel Arc Pro B70: 2× Capacity with strong Long Context Throughput Gains」，Hugging Face Blog，2026-08-06，<https://huggingface.co/blog/kushal-mittal/fp8-kvcache-intel-b70> | 〔部分〕 |
| A9 | R. Turganbay，「Unlocking Longer Generation with Key-Value Cache Quantization」，Hugging Face Blog，2024-05-16，<https://huggingface.co/blog/kv-cache-quantization> | 〔部分〕 |
| A10 | X. Liu, Z. Tang, H. Chen 等，「Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression」，arXiv 2502.01941（ICML 2026） | 〔摘要〕 |
| A11 | W. Gao, X. Zhou, P. Sun, T. Zhang, Y. Wen，「Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving」，MLSys 2025，arXiv 2503.24000 | 〔摘要〕 |

**本專案內部（數字的出處）**

| # | 檔案 |
|:--|:--|
| P1 | `results/m10_longctx_mi300x/analysis/tables_longctx.md`（§1、§2；run 列在檔頭） |
| P2 | `results/RUNLOG_MI300X.md`：Milestone 7e（F3）、10.8、10.9、10.10、14.4(d)、:155、:176、:321 |
| P3 | `docs/research_20261010_followup/F3_dequant_fp8.md`（§4.6 容量表） |
| P4 | `docs/precision_20261013/01_advisor_feedback.md` §3；`05_QA.md` §3–§4；`cards/P04_vllm_fp8_calibration.md` |
| P5 | `docs/precision_20261013/08_report_mi300x.md` §0、§2.9 |
| P6 | `docs/stage3_20261019/environment_snapshot.md`、`stage2_factcheck_and_toolbox.md` |

---

## 9. 本機存檔、沒做的事、限制

**存檔**（`/mlsteam/data/tiara/papers/stage3_L6/`，總共 <1 MB，沒有下載任何 PDF——需要的段落都用 arXiv HTML 讀）：
- `charai.html`：Character.AI 部落格的 Wayback 存檔
- `amd_mlperf41_insights.html`：M2
- `mlperf/inference_rules.adoc`：M1，以及 `mlperf/nv_v{4.1,5.0,5.1}_*README*.md`：M3
- `repos/FlashMLA_README.md`、`repos/FlashMLA_README_ba89a34.md`：O7
- `repos/trtllm_quantization.md`、`repos/trtllm_kv-cache-compression.md`、`repos/trtllm_quant_mode.py`：R10
- `repos/sglang_*.py`：R8
- `repos/llama-context.cpp`：R12
- `rocm/vllm-optimization.html` 等：R16
- `configs/*.json`：§4 的 HF config
- §4 的計算腳本（`kvcalc.py`）、欄位列印腳本（`showcfg.py`）在本 session 的 scratchpad，沒有放進 repo

**沒做的事**
- 沒開 GPU，所以 vLLM 原生 FP8／INT8 KV 在我們題目上的答對率、FP8＋OffloadingConnector、TurboQuant 在 MI300X 上能不能跑，全部是 NOT_MEASURED／NOT_VERIFIED。
- MLPerf 只查了 Llama2-70B（v4.1／5.0／5.1）與 DeepSeek-R1（v5.1）的 NVIDIA README，以及 AMD v4.1；Llama3.1-405B 長上下文題的 KV dtype：NOT_FOUND。
- Gemini、OpenAI、Anthropic、Meta、阿里、Moonshot 的生產 KV 精度：搜過，NOT_FOUND。

**限制**
- 〔部分〕的內容由 WebFetch 的小模型依欄位抽取，**不是逐句人工核對**；圖上讀的數字（例如 NVIDIA 圖 5）是工具讀圖的近似值。
- D-Quant、Kitty 是預印本；「Length, Not Bit Width」是單一作者的 workshop poster；Intel、NVIDIA、AMD、vLLM、Google 的數字都是自測。
- 本檔是 self-review，不是 cross-model review。
