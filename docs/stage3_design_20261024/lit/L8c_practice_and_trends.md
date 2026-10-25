# L8c — KV 精度實務、分層產業趨勢、模型架構與硬體事實（2026-10 現況）

- **日期**：2026-10-24（所有網頁的存取日期都是這天）
- **執行者**：Claude（文獻／實務調查 subagent）。只搜尋與閱讀，沒有跑任何 GPU 程式，也沒有改動 repo 的其他檔案
- **查法**：約 13 次 WebSearch、約 30 次 WebFetch，再用 `gh api`／`curl` 直接讀 GitHub 原始碼、PR 與 issue 內文、HF `config.json`、HF API 的參數量，以及 arXiv API。PDF 用 `pdftotext -layout` 讀相關的節與表，包括 DeepSeek-V3.2-Exp 報告、KIVI、KVQuant。原始碼都標了 commit 或日期：vLLM main @`30aded62d4`（2026-10-22）與 tag v0.32.0（2026-10-18）；SGLang main @`7dc2930585`（2026-10-13）；MLPerf `inference_policies` master @`d3eba2f210`（2026-08-20）
- **審查方式**：**這是 self-review，不是 cross-model review。** 引用與計算都由同一個模型自己核對，沒有找獨立的審查者
- **來源標記**：〔官方文件〕〔原文〕〔部落格〕（廠商或專案的官方部落格）〔程式碼〕（原始碼、PR、issue）〔搜尋摘要〕（只看到搜尋引擎的摘要，沒有打開原頁）〔鏡像〕（官方 repo 有存取限制，config 改從等值的鏡像取得）
- **數字規則**：文中每個數字都來自下列兩種之一：(a) 引用的來源；(b) 我用 config 值算出來的結果，公式寫在表內。查不到的寫 NOT_FOUND，無法驗證的寫 NOT_VERIFIED

---

## 0. 結論

- **FP8 KV 是主流的「可選項」，但多數引擎的預設不是 FP8。** vLLM 的預設是 `auto`，也就是跟模型同型別。唯一的例外是 DeepSeek-V3.2，vLLM 對它預設用 fp8 [R1]。vLLM 官方部落格（2026-04）寫的是 FP8 KV「已可作為多數長上下文部署的預設起點」[R7]。DeepSeek 的推論碼註明實際部署就用 FP8 KV [R17]。MLPerf 規則把 KV 當作 activation 處理，可以量化但不可以剪枝 [R20]。NVIDIA 的 v5.1 DeepSeek-R1 投稿與 AMD 的 Llama2-70B 投稿都用了 FP8 KV [R21][R22][R23]。ROCm 從 6.2 版開始支援 vLLM 的 FP8 KV [R24]。
- **FP8 並不是處處安全。** NVIDIA 的文件明確警告 Qwen2.5-7B-Instruct 在 FP8 KV 下準確度明顯下降，原因是某些層的 KV 數值很大 [R27][R28]。這和我們量到「Qwen2.5 少數 K channel 在約 415 附近」的現象一致。另外，Hopper 上的 FP8 FA3 曾經讓 128K NIAH 從 91% 掉到 13%，後來用兩段式累加修回到 89% [R7]。
- **4-bit KV 在 2026 年已經進入實務，但有條件。**
  - TensorRT-LLM 有 NVFP4 KV，硬體只支援 Blackwell 的 sm100/103 [R9]。
  - vLLM 的 v0.32 已經有 `nvfp4`、`int4_per_token_head`，以及 TurboQuant 的 3 到 4 bit 選項 [R1][R4][R5]。
  - SGLang 有 `nvfp4` 與 `fp4_mx_block16` [R14]。
  - LMDeploy 有 online INT4（`quant_policy=4`）[R15]。
  - **DeepSeek-V4.1-Flash 把 FP4 主 KV 寫進模型設計**（E2M1 加上每 16 個 channel 一個 E4M3 scale），並且把 KV 持久化到 SSD [R19]。
- **低於 4-bit（2 到 3 bit）仍是實驗性的選項，懸崖有文件記錄。**
  - KIVI 的 naive 2-bit 讓 Llama-2-7B 的 GSM8K 從 13.50 掉到 0.83 [R25 Table 3]。
  - KVQuant 的 int2 讓 LLaMA-7B 的 PPL 從 5.68 暴增到 11779 [R26 Table 1]。
  - vLLM TurboQuant 3-bit 讓 Qwen3-4B 的 GSM8K 從 0.900 掉到 0.720 [R5]。
- **「需要原生硬體或付解壓成本」這點有直接證據。** vLLM PR 指出 SM90（H100/H200）沒有 FP4 轉換指令，所以 NVFP4 只能用軟體反量化，走 fa2 路徑 [R6]。NVIDIA 的 NVFP4 KV 在進 attention 之前要先反量化成 FP8 [R8]。TurboQuant 的 decode 吞吐只剩 baseline 的 65% 到 84% [R5]。
- **最關鍵的發現是「INT4 放慢速層」已經是業界作法，不是空白。** TensorRT-LLM 的「NVFP4 cold-page」功能只在 KV 頁位於 Host 或 Disk 時才編碼成 NVFP4，GPU 上維持原本的精度；KV 頁離開 GPU 時量化，回來時反量化 [R11][R12]。它的限制是只支援 SM100/103/107，而且觸發只看記憶體壓力，不看 GPU 負載 [R11][R12]。所以學生原句的「定位」是對的，但不能當作貢獻。差異化必須放在兩點：在 H200（沒有 FP4）上的 INT4 路徑，以及「壓縮、丟棄、重算」的決策本身。
- **原句判定：方向大致正確，但有三處需要修正。** (1)「低於 4 bit 才保守」的界線要改寫：4-bit 本身已經部分上線，只是綁定 Blackwell FP4 或冷層。(2) FP8 要補上模型相依的失敗例（Qwen2.5）。(3) 必須點名 TRT-LLM cold-page 是 prior art。
- **修正後、給指導教授的句子（附來源編號）：**
  > 「FP8 KV 已是各主流引擎的標準選項（vLLM、SGLang、TensorRT-LLM、ROCm），DeepSeek-V3.2 部署即用 FP8，MLPerf 規則允許 KV 依 activation 規則量化，NVIDIA 與 AMD 的投稿也都用了 FP8 KV [R1][R7][R17][R20][R21][R23][R24]。但 FP8 有模型相依的失敗：NVIDIA 文件警告 Qwen2.5-7B 在 FP8 KV 下明顯掉分，這與我們量到的 K 偏移一致 [R27]。4-bit KV 在 2026 年已部分進入實務，但綁定 Blackwell 的原生 FP4（TensorRT-LLM NVFP4、DeepSeek-V4.1 的 FP4 主 KV）[R9][R19]；在 H200（SM90）上因為沒有 FP4 轉換指令，只能付軟體反量化的成本 [R6]。低於 4-bit 仍屬實驗選項，已有文件記錄品質懸崖 [R25][R26][R5]。TensorRT-LLM 已經把 NVFP4 只用在 host/disk 的冷頁 [R11][R12]，所以『INT4 放慢速層』本身是業界作法；我們要研究的是：在沒有原生 FP4 的 H200 上，何時該壓縮、何時丟棄重算、何時保留原精度。」
- **對計畫最重要的三個趨勢**（詳見 §5）：
  1. 4-bit 進入冷層，而且由硬體綁定。
  2. 所有主流引擎都已經有多層 offload，但這次沒找到任何系統依 GPU 負載決定「壓縮、丟棄或重算」。
  3. 新架構（MLA、hybrid linear、Mamba2、SWA）讓每 token 的 KV 縮小 4 到 16 倍，同時多出固定大小的 state。DeepSeek-V4.1 甚至把「restore 時重算 SWA」設計進模型本身。

---

## 1. Q1 — KV 精度實務對照表

| 系統 | KV dtypes | 預設 | 校準／scale | 準確度證據 | 來源 |
|---|---|---|---|---|---|
| **vLLM**（main @30aded62d4；v0.32.0 相同） | `auto, float16, bfloat16, fp8, fp8_e4m3, fp8_e5m2, fp8_inc, fp8_ds_mla, nvfp4_ds_mla, turboquant_k8v4, turboquant_4bit_nc, turboquant_k3v4_nc, turboquant_3bit_nc, ultraquant_4bit, int4_per_token_head, int8_per_token_head, fp8_per_token_head, nvfp4, nvfp4_4over6` 〔程式碼〕 | `"auto"`，即模型的 dtype。help 原文：「Some models (namely DeepSeekV3.2) default to fp8」〔程式碼〕 | FP8 不校準時 scale 全部為 1.0；文件把 llm-compressor 標為「[Recommended]」，支援 per-tensor 與 per-attention-head（per-head 只有 Flash Attention backend 有）〔官方文件〕。`*_per_token_head` 系列在 runtime 動態計算 scale 〔程式碼〕。TurboQuant 不需要離線校準（WHT 旋轉加 Lloyd-Max）〔程式碼〕 | 見下方「vLLM 的準確度證據」 | [R1][R2][R3][R4][R5][R6][R7] |
| **TensorRT-LLM** | 新版 features 頁列出「FP8 KV Cache」與「NVFP4 KV Cache」；舊版 precision 頁有 `INT8_KV_CACHE` 與 `FP8_KV_CACHE` 兩個 flag 〔官方文件〕 | 不是預設。要用 `KvCacheConfig(dtype='fp8')` 或 `dtype='nvfp4'` 開啟 〔官方文件〕 | NVFP4 KV「需要 ModelOpt 離線量化」，而且只能搭配 FP8 的權重與 activation 〔官方文件〕。硬體表：FP8 KV 從 Ampere 到 Blackwell 全部支援；NVFP4 KV 只支援 sm100/103 〔官方文件〕 | 見下方「TensorRT-LLM 的準確度證據」 | [R8][R9][R10][R11][R12][R13] |
| **SGLang**（main @7dc2930585） | `auto, fp8_e5m2, fp8_e4m3, mxfp8, bf16, bfloat16, nvfp4, fp4_mx_block16` 〔程式碼〕 | `"auto"`。checkpoint 的 `kv_cache_quant_algo` 若為 FP8 就自動用 fp8_e4m3 〔程式碼〕 | `nvfp4` 與 `fp4_mx_block16` 都需要 CUDA 12.8+ 與 PyTorch 2.8+；`mxfp8` 只在 FA4 backend 可用 〔程式碼〕 | 選 FP4 時程式會警告「KV Cache might lead to an accuracy drop!」〔程式碼〕。沒找到量化數據（NOT_FOUND） | [R14] |
| **LMDeploy** | `quant_policy=4`（int4）與 `=8`（int8）；`=42` 是 TurboQuant，只在 PyTorch engine 〔官方文件〕 | 需要使用者設定才會開啟（opt-in）〔官方文件〕 | online、data-free，per-head、per-token 非對稱量化；支援 sm70 以上的 NVIDIA GPU 〔官方文件〕 | 見下方「LMDeploy 的準確度證據」 | [R15] |
| **DeepSeek-V3.2 / V3.2-Exp** | 部署用 FP8 KV 〔程式碼〕 | 實際部署就是 FP8 | 每 token、每層 656 B：512 個 FP8 NoPE 值，加 4 個 float32 scale（每 128 個值一個），再加 64 個 BF16 RoPE 值（RoPE 部分刻意不量化）〔程式碼〕。lightning indexer 的 K cache 是 `float8_e4m3fn`，另有 float32 block scale（block_size 128）〔程式碼〕 | 技術報告 §1「Prototype of DSA」寫 indexer「can be implemented in FP8」〔原文〕。推論碼註解「we use fp8 kv cache in actual deployment」〔程式碼〕 | [R16][R17][R18] |
| **DeepSeek-V4.1-Flash**（2026） | FP4 主 KV（E2M1，每 16 個 channel 一個 E4M3 scale）〔官方文件〕 | 是模型設計本身 | FlashMLA 註明 SWA 的 KV 預期用 FP8，CA（compressed attention）的 KV 預期用 FP4 〔程式碼〕。FlashMLA 在 2026-09-30 移除了 Hopper 支援 〔程式碼〕 | 全域 KV 為 890 B/token，約是 V4-Flash 的 1/4 〔官方文件〕。品質比較 NOT_FOUND | [R18][R19] |
| **MLPerf Inference**（rules master @d3eba2f） | 規則不指定格式 | 不適用 | FAQ 原文：KV 項目「should be handled in the same way as the activations」，可以量化、不可以剪枝；CLOSED 組只能用官方提供的校準集 〔官方文件〕 | 精度門檻：Llama2-70B 為 FP32 的 99.9%；Llama3.1-8B 為 99% 與 99.9% 兩檔；DeepSeek-R1 與 Llama3.1-405B 為 FP16 的 99%；GPT-OSS-120B 為 83.13% 的 99% 〔官方文件〕。投稿實況見下方「MLPerf 投稿實況」 | [R20][R21][R22][R23] |
| **AMD ROCm / MI300X** | `fp8`（=fp8_e4m3）〔程式碼〕 | 需要使用者開啟 | ROCm 6.2 起可以用 `--kv-cache-dtype fp8`；token 容量從約 1,262,352 增加到約 2,524,704 〔部落格〕 | KV 段落沒有準確度聲明 〔部落格〕 | [R1][R24] |
| **INT4 或 INT2 預設上線的系統** | 只找到 DeepSeek-V4.1-Flash 的 FP4（4-bit）主 KV 是「預設」；**沒有找到任何服務商預設用 INT4 或 INT2 KV（NOT_FOUND）** | 不適用 | 不適用 | 見下方「低於 4-bit 的懸崖」 | [R19][R25][R26][R5] |

**vLLM 的準確度證據**
- TurboQuant PR 在 Qwen3-4B 上的 GSM8K（baseline 為 0.900）：k8v4 0.860、t4nc 0.840、k3v4nc 0.780、t3nc 0.720。NIAH 全部 100%。decode 吞吐為 baseline 的 65% 到 84%（scenario 為 short-decode、mixed、decode-heavy 與 high-load）〔程式碼〕[R5]。
- INT4 per-token-head PR 自報 GSM8K 0.912，但 PR 內文沒寫是哪個模型，所以 NOT_VERIFIED 〔程式碼〕[R4]。
- FP8 部落格：Llama-3.3-70B 在 MRCR 上恢復 baseline AUC 的 97% 到 98%；修正前，Hopper 的 NIAH 從 91% 掉到 13%，修正後回到 89% 〔部落格〕[R7]。

**TensorRT-LLM 的準確度證據**
- NVIDIA 部落格（2025-12-08）：NVFP4 相對 BF16 與 FP8「less than 1%」的準確度損失。Qwen3-480B 的 Ruler 64K：FP8 約 95.5%，NVFP4 約 94.6%。NVFP4 在 attention 計算前要先反量化成 FP8 〔部落格〕[R8]。
- DeepSeek-R1 調校指南：FP8 KV 相對 BF16「does not introduce obvious accuracy drop」〔部落格〕[R13]。

**LMDeploy 的準確度證據**（OpenCompass，三個數字依序是 fp16 / int8 / int4）〔官方文件〕[R15]
- internlm2-chat-7b 的 GSM8K：70.13 / 69.75 / 66.87。
- llama2-7b-chat 的 TriviaQA：56.09 / 56.13 / 53.71。
- 其他多數 int4 的變化在 1 分以內。
- 吞吐：llama2-7b 的 int4 為 fp16 的 1.39 倍 RPS。

**MLPerf 投稿實況**
- NVIDIA v5.1 的 DeepSeek-R1 把 KV「quantized to FP8 precision」〔部落格〕[R21]。
- AMD MI300X v5.0 的 Llama2-70B checkpoint 把「KV Cache Entries」列為量化張量 〔官方文件〕[R22]。
- AMD v4.1 部落格原文：「implemented FP8 KV cache」〔部落格〕[R23]。

**低於 4-bit 的懸崖**
- KIVI Table 3：naive 2-bit（K、V 都 per-token）讓 Llama-2-7B 的 GSM8K 從 13.50 掉到 0.83，KIVI-2 則是 12.74。KIVI Table 8：Llama-3-8B 的 LongBench 平均從 45.21 降到 44.37（KIVI-2），但 LCC 從 57.00 掉到 50.84 〔原文〕[R25]。
- KVQuant Table 1（LLaMA-7B 的 PPL，baseline 5.68）：int3 為 10.87、int2 為 11779、KVQuant-2bit 為 7.23、KVQuant-2bit-1% 為 6.01 〔原文〕[R26]。

**FP8 的模型相依失敗**
- TensorRT-Edge-LLM 文件的 Warning 段：Qwen2.5-7B-Instruct 與 Qwen2.5-VL-7B「exhibit significant accuracy degradation with FP8 KV cache」。原因寫的是「the last layer has very large values」。對應的 issue 是 NVIDIA/TensorRT-LLM#4218，標題為「FP8 KV Cache is Broken for Qwen2.5」〔官方文件〕〔程式碼〕[R27][R28]。
- 〔判讀〕文件歸因於「最後一層」，我們則量到「少數 K channel 有固定偏移」。兩者是相容的描述，但目前沒有來源指明是 channel 級的偏移，這一點 NOT_VERIFIED。

---

## 2. Q2 — 產業分層趨勢（2025–2026）

| 系統 | 分層 | 是否壓縮 | restore 時能否重算（類似 Cake） | 是否考慮負載 | 來源 |
|---|---|---|---|---|---|
| **NVIDIA Dynamo KVBM**（docs v1.6.0） | G1 GPU、G2 host CPU、G3 disk、G4 object storage（S3 相容）〔官方文件〕 | 設定文件沒提到 〔官方文件〕 | 沒提到；只有 `onboard.mode`（inter 或 intra）〔官方文件〕 | offload 篩選有 `presence` 與 `presence_lfu`（`min_lfu_count` 預設 8），以及 prefix priority。都不看 GPU 負載 〔官方文件〕 | [R29] |
| **NVIDIA Inference Context Memory Storage / BlueField-4**（CES，2026-01-05） | 以 BlueField-4 為基礎的「AI-native storage」，用來擴充 GPU 的 context memory。整合 DOCA、NIXL、Dynamo，網路是 Spectrum-X RDMA；合作夥伴產品預計 2026 下半年上市 〔官方文件〕 | 新聞稿沒提到 | 沒提到 | 沒提到。宣稱 tokens/s 最多 5 倍，相對傳統 storage 的能效最多 5 倍，但沒有給出基準設定 〔官方文件〕 | [R30] |
| **TensorRT-LLM KVCacheManagerV2 + cold-page 壓縮** | GPU、Host、Disk 〔官方文件〕 | **只在 Host/Disk 層用 NVFP4**，GPU 上保持原 dtype；KV 頁離開 GPU 時量化，回來時反量化；GDN、SSM、DSA 的附屬 buffer 無損保留 〔官方文件〕 | 沒有；重用的頁直接跳過 prefill 〔部落格〕 | 只在記憶體壓力下觸發，沒有可調的負載政策 〔部落格〕 | [R11][R12] |
| **LMCache** | CPU RAM、Local storage、GDS、Redis/Valkey、Mooncake、InfiniStore、NIXL、Weka、S3、3FS 等 〔官方文件〕 | 有 CacheGen 可選：`remote_serde` 預設是 `naive`，可改成 `cachegen`；multiprocess 模式下 CacheGen「尚未實作」〔搜尋摘要〕 | 這次沒查到（NOT_VERIFIED） | 沒找到 | [R31][R32] |
| **Mooncake Store** | 分散式 DRAM pool，可以從 RAM offload 到 SSD，支援多副本 〔程式碼〕 | 設計文件沒提到 | 沒提到 | 多副本是為了緩解「access pressure」的熱點；不看 GPU 負載 〔程式碼〕 | [R33] |
| **llm-d tiered prefix cache** | GPU HBM、CPU RAM、共享的 RWX filesystem（vLLM `TieringOffloadingSpec`）；另可用 Mooncake（DRAM+SSD）、LMCache、SGLang HiCache 〔官方文件〕 | 沒提到 | 沒有決策機制；只寫了經濟條件「A load has to finish faster than the prefill it replaces」〔官方文件〕 | **路由層**（EPP）會同時考慮 prefix 與負載，每層有自己的 prefix index；但這不是分層或壓縮的決策 〔官方文件〕 | [R34] |
| **SGLang HiCache** | L1 GPU、L2 host、L3 分散式 storage（Mooncake、3FS、NIXL、AIBrix、file）〔官方文件〕 | 沒提到 | 部分：`best_effort` 預取策略會在 GPU 可以開始 prefill 時立即結束預取，沒抓到的部分就重算 〔官方文件〕 | 寫入策略有 write_through、write_through_selective（看存取頻率）、write_back；預取門檻預設 256 tokens；不看 GPU 負載 〔官方文件〕 | [R35] |
| **vLLM OffloadingConnector + TieringOffloadingSpec**（2026-05-13 合併，#40020） | CPU 主層，加上次層 `fs`、`p2p`、`obj`；主層 eviction 可選 `lru` 或 `arc` 〔程式碼〕 | 在 `kv_offload` 目錄搜尋「compress」，結果 0 筆（NOT_FOUND）〔程式碼〕 | 只在 miss 或失敗時重算：`kv_load_failure_policy` 可設 `"recompute"` 或 `"fail"`（預設 fail）〔程式碼〕 | 有**I/O 背壓**：用 `EMABackpressureDetector` 觀察 store latency，搭配 `DropStorePolicy` 或 `ThrottledDropPolicy` 丟掉寫入；看的是 storage I/O，不是 GPU 負載 〔程式碼〕 | [R36] |
| **WEKA Augmented Memory Grid** | 以 NVMe 為底的「Token Warehouse」，資料路徑是 GDS 加 RDMA；整合 Dynamo 與 NIXL plugin，也有支援 TRT-LLM 與 LMCache 的開源專案 〔官方文件〕 | 沒提到 | 宣稱消除重複的 prefill；Nebius 案例為「2.4x more requests」與「3.2x fewer tokens recomputed」〔官方文件〕 | 沒提到。TTFT 宣稱有 4 到 20 倍，以及 105K prompt 從 23.97 s 降到 0.58 s 〔搜尋摘要〕 | [R37] |
| **CXL 類論文** | Beluga（SIGMOD'26）：用 CXL switch 建立 GPU 與 CPU 共享的記憶體池，整合 vLLM，相對 RDMA 方案 TTFT 降 89.6%、吞吐 7.35 倍 〔原文〕。PNM 論文（arXiv 2511.00321）：在 CXL 記憶體內的 PNM 做 token page 選擇，最高 21.9 倍吞吐 〔原文〕 | 摘要沒提到 | 摘要沒提到 | 摘要沒提到 | [R38][R39] |
| **GPUDirect Storage 用於 KV** | Dynamo KVBM 的 disk 層有 `USE_GDS` 與 `DYN_KVBM_NIXL_BACKEND_GDS`；LMCache 有 GDS backend；WEKA AMG 走 GDS 〔官方文件〕 | 不適用 | 不適用 | 不適用 | [R29][R31][R37] |
| **（模型端）DeepSeek-V4.1-Flash** | 持久化 KV 放在 SSD；**SWA 的 KV 不寫入 SSD** 〔官方文件〕 | FP4 主 KV 〔官方文件〕 | **有：「SWA Bounded Replay」只重播最近 n_win 個 token 來重建 SWA KV**（n_win 值 NOT_FOUND）；持久化的 KV 約為 V4-Flash 的 1/8 〔官方文件〕 | 沒提到 | [R19] |

〔判讀〕整張表看下來有三個共同點：
1. 壓縮只在 TRT-LLM（冷頁 NVFP4，限 Blackwell）與 LMCache（CacheGen，可選）出現。
2. 「重算」在各系統裡只是 miss 或失敗時的後備（vLLM、SGLang best_effort），或是模型設計的一部分（DeepSeek-V4.1）。沒有任何系統依成本或負載主動選擇重算。
3. 「負載」只出現在三處：I/O 背壓（vLLM 丟寫入）、路由（llm-d）、存取頻率（Dynamo、SGLang）。**這次沒找到任何系統依 GPU 負載決定要壓縮、丟棄還是重算。**

---

## 3. Q3 — 架構對每 token KV bytes 的影響

**計算規則**

- 一般 attention：每 token 的 bytes 是 `L × kv_heads × head_dim × 2 (K,V) × 2 B`（BF16）。
- MLA：`L × (kv_lora_rank + qk_rope_head_dim) × 2 B`。
- linear 或 SSM 層：狀態大小固定，不隨 token 數成長，所以列「每序列元素數」。這些狀態的實際 dtype 這次沒查到，標 NOT_VERIFIED；列出的 MiB 是假設 fp32 的參考值。
- `head_dim` 沒寫在 config 時，用 `hidden_size / num_attention_heads`。
- 下表的數字都是我依 config 值計算的 〔判讀〕；config 本身是〔官方文件〕。

**「放得進 H200」的判準** 〔判讀〕：權重要 ≤ 0.8 × 141 GB × 卡數。141 GB 取自 [R45]，參數量取自 HF API 的 safetensors 統計 [R44]。

| 模型 | attention 類型 | config 值 | KV bytes/token @BF16（公式） | 開放權重？放得進 1–4× H200？ | 來源 |
|---|---|---|---|---|---|
| **Llama-3.1-8B-Instruct** | GQA | L=32, kv=8, hd=128 | 32·8·128·2·2 = **131,072 B（128 KiB）** | 有（gated）；8.03B → 1 張 | [R40a]〔鏡像〕[R44] |
| **Qwen2.5-7B-Instruct-1M** | GQA（config 含 DCA 設定） | L=28, kv=4, 3584/28=128；`dual_chunk_attention_config` chunk 262144 | 28·4·128·2·2 = **57,344 B（56 KiB）** | 有；7.62B → 1 張 | [R40b][R44] |
| Qwen3-8B（GQA 對照） | GQA | L=36, kv=8, hd=128 | 36·8·128·4 = **147,456 B（144 KiB）** | 有；8.19B → 1 張 | [R40c][R44] |
| **DeepSeek-V3.2(-Exp)** | MLA + DSA（lightning indexer，top-k=2048） | L=61, kv_lora_rank=512, rope=64；indexer 64 heads × 128 | MLA：61·(512+64)·2 = **70,272 B**。indexer K 以 BF16 等價算為 61·128·2 = 15,616 B。**實際部署格式**：61·656 = 40,016 B，加上 indexer 的 61·(128 FP8 + 4 B scale) = 8,052 B，合計約 48,068 B | 有；685.4B，FP8 約 685 GB，**超過 4×141 GB**，所以 4 張以內放不下 | [R40d][R17][R18][R44] |
| **Kimi K2** | MLA | L=61, 512+64, 64 heads | 61·576·2 = **70,272 B** | 有；1.03T，放不下 | [R40e][R44] |
| DeepSeek-V2-Lite-Chat（MLA 的小型代表） | MLA | L=27, 512+64 | 27·576·2 = **31,104 B** | 有；15.7B → 1 張 | [R40f][R44] |
| **Kimi-Linear-48B-A3B** | hybrid：7 層 MLA 加 20 層 KDA | full_attn_layers=[4,8,12,16,20,24,27]；KDA num_heads=32, head_dim=128 | MLA 部分：7·(512+64)·2 = **8,064 B**。論文是否 NoPE、因而 rope 部分不存，NOT_VERIFIED，這個值是上界。KDA 狀態為 20·32·128·128 = 10,485,760 個元素／序列（以 fp32 計約 40 MiB）。論文宣稱 KV 用量最多降 75% 〔原文〕 | 有；49.1B → BF16 1 張（偏緊） | [R40g][R41][R44] |
| **Qwen3-Next-80B-A3B** | hybrid：12 層 gated attention 加 36 層 Gated DeltaNet（每 4 層 1 層 full） | full_attention_interval=4, kv=2, hd=256；linear 部分 v-heads 32、k-heads 16、dk=dv=128、conv 4 | 12·2·256·2·2 = **24,576 B**。GDN 狀態為 36·32·128·128 = 18,874,368 個元素，另有 conv 884,736 個元素 | 有；81.3B → BF16 2 張、FP8 1 張 | [R40h][R44] |
| Qwen3.5-35B-A3B（2026-02） | hybrid：10 層 full 加 30 層 GDN | 40 層，interval 4，kv=2, hd=256 | 10·2·256·4 = **20,480 B**；GDN 狀態 15,728,640 個元素 | 有；35.95B → 1 張 | [R40i][R44] |
| Qwen3.5-27B（2026-02） | hybrid：16 層 full 加 48 層 GDN | 64 層，kv=4, hd=256，v-heads 48 | 16·4·256·4 = **65,536 B**；GDN 狀態 37,748,736 個元素 | 有；27.8B → 1 張 | [R40j][R44] |
| **Nemotron-Nano-9B-v2** | hybrid：Mamba2 加 4 層 attention | pattern 共 56 層：4 個 `*`、27 個 `M`、25 個 `-`；kv=8, hd=128；mamba 128 heads × 80，state 128 | 4·8·128·4 = **16,384 B**；Mamba2 狀態 27·128·80·128 = 35,389,440 個元素 | 有；8.89B → 1 張；max_pos 131072 | [R40k][R41][R44] |
| Nemotron-H-47B-Base-8K | hybrid：Mamba2 加 5 層 attention | pattern 共 98 層：5 個 `*`、45 個 `M`；kv=8, hd=128 | 5·8·128·4 = **20,480 B**；狀態 188,743,680 個元素 | 有；46.8B → 1 張；**但 max_pos 只有 8192**，不適合長上下文 | [R40l][R41][R44] |
| **MiniMax-M1-80k** | hybrid：70 層 lightning（linear）加 10 層 softmax | attn_type_list 中 0 有 70 個、1 有 10 個；kv=8, hd=128 | 10·8·128·4 = **40,960 B**；lightning 狀態 70·64·128·128 = 73,400,320 個元素 | 有；456B，放不下 | [R40m][R41][R44] |
| MiniMax-M2 | **全部是 full attention**，M2 退回了 full | 62 層的 attn_type 全為 1；kv=8, hd=128 | 62·8·128·4 = **253,952 B** | 有；228.7B，FP8 需約 3 張（超過 ≤120B 的條件） | [R40n][R44] |
| **gpt-oss-120b** | SWA（window 128）與 full 逐層交替 | 36 層，18 層 sliding、18 層 full；kv=8, hd=64；MoE 權重 MXFP4 | full 層：18·8·64·4 = **36,864 B/token**。SWA 層有上限：18·8·64·4·128 = 4.5 MiB／序列 | 有；117B；model card 寫「fit into a single 80GB GPU」→ 1 張 | [R40o][R43][R44] |
| gpt-oss-20b | 同上 | 24 層，12 層 sliding、12 層 full | 12·8·64·4 = **24,576 B**；SWA 上限 3 MiB | 有；20.9B → 1 張 | [R40p][R44] |
| **Gemma 3 27B** | 5:1 local/global，local window 1024 | 62 層，sliding_window=1024，pattern=6，kv=16, hd=128 | 依 pattern=6 推算有 10 層 global 與 52 層 local（分層 NOT_VERIFIED）。global：10·16·128·4 = **81,920 B**；local 上限 52·16·128·4·1024 = 416 MiB／序列 | 有（gated）；27.4B → 1 張 | [R40q]〔鏡像〕[R44] |
| **MiniCPM4-8B** | InfLLM-v2 trainable sparse：KV 全部保存，只讀一個子集 | L=32, kv=2, 4096/32=128；`sparse_config` 要手動加進 config（topk=64、block 64、window 2048、dense_len 8192）〔官方文件〕 | 32·2·128·4 = **32,768 B**（KV 照樣全存）。model card 寫 128K 時每 token 只需和「less than 5% of tokens」計算相關性 | 有；8.19B → 1 張；原生 32K，用 LongRoPE 驗證到 131,072 | [R40r][R42][R44] |
| NSA（DeepSeek） | native sparse attention | 不適用 | 不適用 | **沒找到官方開放權重**（NOT_FOUND），只有論文 | [R41] |
| MoBA（Moonshot） | block sparse attention | 不適用 | 不適用 | **沒找到原生採用 MoBA 的開放權重模型**（NOT_FOUND），只有論文 | [R41] |
| DeepSeek-V4.1-Flash（趨勢參考） | CSA2 sparse，causal encoder-decoder，40 層 | 不適用 | model card：用 FP4 時全域 KV 為 **890 B/token** 〔官方文件〕 | 有（MIT）；763B，放不下 | [R19][R44] |

〔判讀〕以 128K context 為例，只算會隨 token 成長的 KV：
- Llama-3.1-8B 是 16.0 GiB。
- Qwen2.5-7B-1M 是 7.0 GiB。
- DeepSeek MLA 是 8.6 GiB。
- gpt-oss-120b 的 full 層是 4.5 GiB。
- MiniCPM4-8B 是 4.0 GiB。
- Qwen3-Next 是 3.0 GiB。
- Nemotron-Nano-9B-v2 是 2.0 GiB。
- Kimi-Linear 的 MLA 層約 0.98 GiB。

同樣 context 下最多相差約 16 倍。hybrid 模型另外有每序列數十到數百 MiB 的固定狀態。這些固定狀態無法「部分重算」：要不就整份載回，要不就從頭重播整段。

---

## 4. Q4 — 硬體事實（每項一行）

- **H200（SXM 與 NVL）**：141 GB、4.8 TB/s、NVLink 900 GB/s、**PCIe Gen5 128 GB/s**；SXM 的 FP8 為 3,958 TFLOPS（含 sparsity），TDP 最高 700 W 〔官方文件〕[R45]。官方頁沒說 128 GB/s 是雙向還是單向，NOT_VERIFIED。
- **H200／H100（SM90）不支援 FP4**：vLLM PR #60620 引 FlashInfer 原始碼，指出 SM90「has no FP4 conversion instructions」，NVFP4 只能軟體反量化，走 fa2 〔程式碼〕[R6]。
- **GH200**：NVLink-C2C 為「900 gigabytes per second (GB/s) of coherent interface」，「7X faster than PCIe Gen5」〔官方文件〕[R46]。
- **GB200**：「900 GB/s NVIDIA® NVLink®-C2C」連到兩顆 Blackwell GPU。是否雙向 NOT_VERIFIED 〔官方文件〕[R47]。
- **Blackwell Decompression Engine**：支援 Snappy、LZ4、Deflate/Gzip，最高 600 GB/s；只有 B200、B300、GB200、GB300 有；B200 的單一 chunk 上限是 4 MiB，條件不符時退回用 SM 解壓 〔官方文件〕[R48]。〔判讀〕這是通用的位元組壓縮，不支援浮點或量化的 KV 格式，所以不能直接拿來做 KV 反量化。
- **MI300X**：192 GB HBM3、5.3 TB/s、PCIe 5.0 x16、8 條 Infinity Fabric link（128 GB/s）；FP8（E5M2、E4M3）峰值 2.61 PFLOPs 〔官方文件〕[R49][R51]。
- **MI355X**：288 GB HBM3E、8 TB/s、PCIe 5.0 x16；MXFP4 峰值 10.1 PFLOPs（CDNA4，gfx950）〔官方文件〕[R50][R51]。〔判讀〕MI355X 有原生 FP4，和 Blackwell 一樣能讓 4-bit KV「不付反量化成本」；MI300X 和 H200 一樣沒有。

---

## 5. 對 H200 研究計畫的含意（〔判讀〕）

1. **〔判讀〕推薦的 H200 模型組合，每個都能單卡跑。** 好處是可以每卡一個 job 平行掃描，而且符合「容量在單卡量」的原則。
   - (a) **Llama-3.1-8B**：GQA 基準，128 KiB/token，延續 3090 的結果。
   - (b) **Qwen2.5-7B-Instruct-1M**：GQA，56 KiB/token，是已知 FP8 會壞的案例 [R27]。
   - (c) **DeepSeek-V2-Lite-Chat**：MLA，30 KiB/token。DeepSeek-V3.2 和 Kimi K2 在 4 張以內放不下，用它代表 MLA。
   - (d) **Qwen3.5-35B-A3B** 或 **Qwen3-Next-80B-A3B**（FP8 權重）：GDN hybrid，3:1，20 到 24 KiB/token 加固定狀態。
   - (e) **gpt-oss-20b 或 120b**：SWA128 與 full 交替，也是 MLPerf 的模型。
   - (f) **Nemotron-Nano-9B-v2**：Mamba2，56 層裡只有 4 層 attention，16 KiB/token。
   - (g) **MiniCPM4-8B**：InfLLM-v2，代表 DSA 那種「全部存、稀疏讀」。
   - 選配：**Kimi-Linear-48B-A3B**（MLA 加 KDA）、**Gemma-3-27B**（5:1 local/global，window 1024）。
   - 各模型在你們 vLLM 版本與 H200 上能不能跑，這次沒有驗證（NOT_VERIFIED），需要先跑 smoke test。
2. **〔判讀〕必須把 TRT-LLM 的 NVFP4 cold-page 當作最接近的 prior art** [R11][R12]。我們的差異點可以放在三處：
   - (i) H200（SM90）沒有 FP4 指令，INT4 或 NVFP4 都要付軟體反量化成本 [R6]，所以「壓縮能不能划算」本身就是一個依硬體而變的問題。這正好接上 Tiara 的 κ 主張。
   - (ii) TRT-LLM 的編碼是靜態的：所有冷頁都壓，只看記憶體壓力；我們要做的是逐位置或逐層，在「壓縮、丟棄、重算」之間做決策。
   - (iii) GPU 負載感知：這次的調查沒有找到任何系統做這件事。
3. **〔判讀〕vLLM v0.32 已經有很多現成的 baseline，不需要自己做**：`int4_per_token_head`（Triton）、`nvfp4`（SM90 走 fa2）、TurboQuant、`TieringOffloadingSpec`（fs、p2p、obj 次層，加 I/O 背壓丟寫入）[R1][R4][R5][R6][R36]。這和我們關心的「寫入當下就決定（write-time）」直接相關：vLLM 已經會在背壓時丟掉寫入，所以「在什麼條件下不寫」的對照組可以直接用它。
4. **〔判讀〕在 H200 上做 FP8 對照時要排除兩個已知陷阱**：
   - (i) 要用含 FA3 兩段式累加修正的版本。部落格的「修正後」是用 v0.19.1 量的 [R7]，用更舊的版本會在長上下文掉分。
   - (ii) FP8 的公平 baseline 應該加入 llm-compressor 的 per-head 校準 [R2]。否則 Qwen2.5 的失敗會被歸因成「FP8 本來就不行」，而不是「沒有校準」。
5. **〔判讀〕hybrid 和 SSM 模型改變了分層的單位。** 固定狀態必須無損保留：TRT-LLM 也把 GDN、SSM 狀態「preserved losslessly」[R11]。而且這些狀態不能逐 token 地部分重算。實驗應該把「attention KV（可壓縮、可部分重算）」和「固定狀態（只能整份搬，或整段重播）」分開記帳。
6. **〔判讀〕MLA 和 sparse 模型讓每 token 的 KV 已經很小，DeepSeek-V3.2 部署時約 48 KB/token [R17][R18]。** 計畫要有一組實驗檢查：KV 小 4 到 16 倍的時候，分層與壓縮的收益還在不在。如果不在，論文主張的適用範圍就要縮到 GQA 類模型，這個結論應該在預先登記時就寫下。
7. **〔判讀〕DeepSeek-V4.1 的「SWA Bounded Replay」是模型端做的「restore 時重算」**，FP4 持久化 KV 則是模型端的「冷層壓縮」[R19]。這兩點證明「重算對比載回」與「冷層低精度」都是活躍的工業方向。可以拿來說明研究問題是真實的；但也代表學界的獨特貢獻必須是**決策準則**，而不是機制本身。
8. **〔判讀〕把 2 到 3 bit 列為延伸實驗，不放進主線。** 文件化的懸崖包括 naive 2-bit GSM8K 從 13.50 掉到 0.83 [R25]，以及 TurboQuant 3-bit 從 0.900 掉到 0.720 [R5]。主線維持 FP8 對 INT4（以及 SM90 上的 NVFP4 軟體路徑），和「放 4-bit KV 時需要硬體支援、否則要付反量化成本」的業界現況一致。

---

## 6. References（存取日期一律為 2026-10-24）

| # | 來源 | URL | 狀態 |
|---|---|---|---|
| R1 | vLLM `vllm/config/cache.py`（CacheDType、cache_dtype 的 help 文字），main @30aded62d4 與 tag v0.32.0 | https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py | 〔程式碼〕已讀原始碼 |
| R2 | vLLM docs「Quantized KV Cache」 | https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/ | 〔官方文件〕已讀（也讀了 GitHub 上的 md 原文） |
| R3 | vLLM docs「Engine Arguments」`--kv-cache-dtype` | https://docs.vllm.ai/en/latest/configuration/engine_args/ | 〔官方文件〕已讀 |
| R4 | vLLM PR #40835「Triton INT4 per-token-head KV cache quantization」（2026-06-24 合併） | https://github.com/vllm-project/vllm/pull/40835 | 〔程式碼〕已讀 PR 內文 |
| R5 | vLLM PR #38479「TurboQuant: 2-bit KV cache compression…」（2026-04-15 合併） | https://github.com/vllm-project/vllm/pull/38479 | 〔程式碼〕已讀 PR 內文與數據表（PR 自報） |
| R6 | vLLM PR #60620「Allow an NVFP4 KV cache on SM90 with fa2」（2026-10-08 合併） | https://github.com/vllm-project/vllm/pull/60620 | 〔程式碼〕已讀；PR 自述沒有在 SM90 實機驗證 |
| R7 | vLLM blog「FP8 KV-Cache and Attention Quantization in vLLM」（2026-04-22） | https://vllm.ai/blog/fp8-kvcache | 〔部落格〕已讀 |
| R8 | NVIDIA Technical Blog「Optimizing Inference … with NVFP4 KV Cache」（2025-12-08） | https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/ | 〔部落格〕已讀；圖上的數值是摘要模型讀圖估的，約略值 |
| R9 | TensorRT-LLM「Quantization」features（支援矩陣） | https://nvidia.github.io/TensorRT-LLM/features/quantization.html | 〔官方文件〕已讀 |
| R10 | TensorRT-LLM「Numerical Precision」 | https://nvidia.github.io/TensorRT-LLM/reference/precision.html | 〔官方文件〕已讀 |
| R11 | TensorRT-LLM「KV Cache Compression」feature（最後更新 2026-10-19，commit 0372799） | https://nvidia.github.io/TensorRT-LLM/latest/features/kv-cache-compression.html | 〔官方文件〕已讀 |
| R12 | TensorRT-LLM tech blog 30「KV Cache Compression in TensorRT LLM」 | https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog30_KV_Cache_Compression_in_TensorRT-LLM.html | 〔部落格〕已讀；標題的 1.64x 與 1.9x 和內文圖表（64%、48%）出自不同設定 |
| R13 | TensorRT-LLM「Best perf practice on DeepSeek-R1」 | https://nvidia.github.io/TensorRT-LLM/latest/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.html | 〔部落格〕已讀 |
| R14 | SGLang `arg_groups/fields/model.py`、`mem_cache/kv_cache_dtype.py`（main @7dc2930585） | https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/arg_groups/fields/model.py ；https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/kv_cache_dtype.py | 〔程式碼〕已讀原始碼 |
| R15 | LMDeploy「INT4/INT8 KV Cache」 | https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html | 〔官方文件〕已讀 |
| R16 | DeepSeek-V3.2-Exp tech report PDF，§1 Architecture，「Prototype of DSA」段 | https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf | 〔原文〕pdftotext 已讀 |
| R17 | DeepSeek-V3.2-Exp `inference/model.py`（Indexer 的 k_cache 是 float8_e4m3fn；MLA 註解寫部署用 FP8；block_size=128） | https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py | 〔程式碼〕已讀 |
| R18 | FlashMLA README：舊 commit ba89a34 有 V3.2 的 656 B 格式；main 有 V4.1 的 FP8 與 FP4 格式，以及 2026-09-30 移除 Hopper 的公告 | https://github.com/deepseek-ai/FlashMLA/blob/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/README.md ；https://github.com/deepseek-ai/FlashMLA | 〔程式碼〕已讀 |
| R19 | DeepSeek-V4.1-Flash model card（Introduction，Architecture 段） | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash | 〔官方文件〕已讀；n_win 值 NOT_FOUND |
| R20 | MLPerf `inference_rules.adoc`（「Weight Definition and Quantization」、Benchmarks 表、FAQ「LLM Benchmarks」），master @d3eba2f | https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc | 〔官方文件〕已讀原文。版本號沒有標在檔案裡 |
| R21 | NVIDIA blog「Blackwell Ultra Sets New Inference Records in MLPerf Debut」（2025-09-09，v5.1） | https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut | 〔部落格〕已讀 |
| R22 | AMD HF card「Llama-2-70b-chat-hf_FP8_MLPerf_V2」（MLPerf v5.0） | https://huggingface.co/amd/Llama-2-70b-chat-hf_FP8_MLPerf_V2 | 〔官方文件〕已讀 |
| R23 | AMD blog「Engineering Insights: Unveiling MLPerf Results on AMD Instinct MI300X」（2024-08-28） | https://www.amd.com/en/blogs/2024/engineering-insights-unveiling-mlperf-results-on.html | 〔部落格〕WebFetch 逾時，改用 curl 抓到原文 |
| R24 | ROCm blog「Enhancing vLLM Inference on AMD GPUs」，「FP8 KV cache」段（2024-09-19） | https://rocm.blogs.amd.com/artificial-intelligence/vllm-optimize/README.html | 〔部落格〕已讀 |
| R25 | KIVI，arXiv 2402.02750v2（Table 1、3、8） | https://arxiv.org/abs/2402.02750 | 〔原文〕pdftotext 已讀 |
| R26 | KVQuant，arXiv 2401.18079v6（Table 1） | https://arxiv.org/abs/2401.18079 | 〔原文〕pdftotext 已讀 |
| R27 | TensorRT-Edge-LLM「FP8 KV Cache」（Warning 段；最後更新 2026-09-29） | https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/FP8KV.html | 〔官方文件〕已讀 |
| R28 | NVIDIA/TensorRT-LLM issue #4218「FP8 KV Cache is Broken for Qwen2.5」（2025-05-12，已關閉） | https://github.com/NVIDIA/TensorRT-LLM/issues/4218 | 〔程式碼〕已讀；issue 的原始回報是 Qwen2.5-Coder 的 fine-tune 模型 |
| R29 | NVIDIA Dynamo「KVBM configuration」（v1.6.0） | https://docs.nvidia.com/dynamo/latest/reference/components/kvbm-configuration.md | 〔官方文件〕已讀 |
| R30 | NVIDIA 新聞稿「BlueField-4 Powers New Class of AI-Native Storage…」（2026-01-05） | https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-BlueField-4-Powers-New-Class-of-AI-Native-Storage-Infrastructure-for-the-Next-Frontier-of-AI/default.aspx | 〔官方文件〕已讀；5x 的基準沒有交代 |
| R31 | LMCache「Storage backends」 | https://docs.lmcache.ai/kv_cache/storage_backends/index.html | 〔官方文件〕已讀（後端清單來自側欄） |
| R32 | LMCache CacheGen、configurations、mp/cachegen | https://docs.lmcache.ai/kv_cache_optimizations/compression/cachegen.html ；https://docs.lmcache.ai/api_reference/configurations.html ；https://docs.lmcache.ai/mp/cachegen.html | 〔搜尋摘要〕沒有打開原頁 |
| R33 | Mooncake Store 設計文件 | https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/store/mooncake-store.md | 〔程式碼〕已讀 md 原文 |
| R34 | llm-d「Tiered prefix cache」well-lit path | https://llm-d.ai/docs/well-lit-paths/foundations/tiered-prefix-cache | 〔官方文件〕已讀 |
| R35 | SGLang「HiCache design」 | https://docs.sglang.io/advanced_features/hicache_design.html | 〔官方文件〕已讀 |
| R36 | vLLM `v1/kv_offload/tiering/{spec,backpressure,factory}.py`、`fs/manager.py`、`config/kv_transfer.py`；PR #40020（2026-05-13） | https://github.com/vllm-project/vllm/tree/main/vllm/v1/kv_offload/tiering ；https://github.com/vllm-project/vllm/blob/main/vllm/config/kv_transfer.py | 〔程式碼〕已讀原始碼 |
| R37 | WEKA Augmented Memory Grid 產品頁；TTFT 數字（4–20x、23.97 s→0.58 s） | https://www.weka.io/product/augmented-memory-grid/ | 產品頁為〔官方文件〕，已讀；TTFT 數字是〔搜尋摘要〕，NOT_VERIFIED |
| R38 | Beluga，arXiv 2511.20172（SIGMOD'26） | https://arxiv.org/abs/2511.20172 | 〔原文〕只讀了摘要頁 |
| R39 | CXL PNM for 1M-token inference，arXiv 2511.00321 | https://arxiv.org/abs/2511.00321 | 〔原文〕只讀了摘要頁 |
| R40a | Llama-3.1-8B-Instruct config（官方 repo 是 gated，回 401，改用鏡像） | https://huggingface.co/unsloth/Llama-3.1-8B-Instruct/blob/main/config.json （官方：https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct） | 〔鏡像〕 |
| R40b | Qwen2.5-7B-Instruct-1M config | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M/blob/main/config.json | 〔官方文件〕 |
| R40c | Qwen3-8B config | https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json | 〔官方文件〕 |
| R40d | DeepSeek-V3.2 與 V3.2-Exp config（兩者相同） | https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json | 〔官方文件〕 |
| R40e | Kimi-K2-Instruct config | https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json | 〔官方文件〕 |
| R40f | DeepSeek-V2-Lite-Chat config | https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json | 〔官方文件〕 |
| R40g | Kimi-Linear-48B-A3B-Instruct config | https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/config.json | 〔官方文件〕 |
| R40h | Qwen3-Next-80B-A3B-Instruct config | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json | 〔官方文件〕 |
| R40i | Qwen3.5-35B-A3B config | https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json | 〔官方文件〕 |
| R40j | Qwen3.5-27B config | https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json | 〔官方文件〕 |
| R40k | NVIDIA-Nemotron-Nano-9B-v2 config | https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/blob/main/config.json | 〔官方文件〕 |
| R40l | Nemotron-H-47B-Base-8K config（另有 8B 的版本） | https://huggingface.co/nvidia/Nemotron-H-47B-Base-8K/blob/main/config.json | 〔官方文件〕 |
| R40m | MiniMax-M1-80k config | https://huggingface.co/MiniMaxAI/MiniMax-M1-80k/blob/main/config.json | 〔官方文件〕 |
| R40n | MiniMax-M2 config | https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/config.json | 〔官方文件〕 |
| R40o | gpt-oss-120b config | https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json | 〔官方文件〕 |
| R40p | gpt-oss-20b config | https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json | 〔官方文件〕 |
| R40q | Gemma-3-27B-it config（官方 repo 是 gated，回 401，改用鏡像） | https://huggingface.co/unsloth/gemma-3-27b-it/blob/main/config.json （官方：https://huggingface.co/google/gemma-3-27b-it） | 〔鏡像〕 |
| R40r | MiniCPM4-8B config | https://huggingface.co/openbmb/MiniCPM4-8B/blob/main/config.json | 〔官方文件〕 |
| R41 | arXiv：NSA 2502.11089、MoBA 2502.13189、Kimi Linear 2510.26692、MiniCPM4 2506.07900、Nemotron-H 2504.03624、MiniMax-M1 2506.13585 | https://arxiv.org/abs/2502.11089 ；https://arxiv.org/abs/2502.13189 ；https://arxiv.org/abs/2510.26692 ；https://arxiv.org/abs/2506.07900 ；https://arxiv.org/abs/2504.03624 ；https://arxiv.org/abs/2506.13585 | 〔原文〕只透過 arXiv API 讀了標題與摘要 |
| R42 | MiniCPM4-8B model card（InfLLM v2 的 sparse_config） | https://huggingface.co/openbmb/MiniCPM4-8B | 〔官方文件〕已讀 |
| R43 | gpt-oss-120b model card | https://huggingface.co/openai/gpt-oss-120b | 〔官方文件〕已讀 |
| R44 | HF API 的 safetensors 參數統計（各模型的 `https://huggingface.co/api/models/<id>`） | https://huggingface.co/api/models/Qwen/Qwen3.5-35B-A3B （其他模型同樣格式） | 〔官方文件〕API 已讀 |
| R45 | NVIDIA H200 產品頁（規格表） | https://www.nvidia.com/en-us/data-center/h200/ | 〔官方文件〕已讀；頁面註明「Preliminary specifications」 |
| R46 | NVIDIA Grace Hopper Superchip 產品頁 | https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/ | 〔官方文件〕已讀 |
| R47 | NVIDIA Multi-Node NVLink Tuning Guide，「The NVIDIA Grace Blackwell Superchip」 | https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/overview.html | 〔官方文件〕已讀 |
| R48 | nvCOMP「Decompression Engine」FAQ | https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html | 〔官方文件〕已讀 |
| R49 | AMD Instinct MI300X 產品頁 | https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html | 〔官方文件〕WebFetch 逾時，改用 curl 抽取規格表 |
| R50 | AMD Instinct MI355X 產品頁 | https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html | 〔官方文件〕WebFetch 逾時，改用 curl 抽取規格表 |
| R51 | ROCm「GPU hardware specifications」（ROCm 10.1.0） | https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html | 〔官方文件〕已讀 |
