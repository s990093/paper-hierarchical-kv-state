# D5_LMCacheHybrid LMCache（multiprocess 模式）對混合模型的支援：Mamba／GDN 狀態與滑動視窗層

- **出處**：LMCache 開源程式碼（系統）。`LMCache/LMCache@7d7ca470472352cef8a4dc284f8481dfba03b923`（dev 分支，2026-10-09）。讀了 `docs/source/mp/hybrid_models.rst`、`docs/design/integration/vllm/hybrid-kv-cache-groups.md`、`lmcache/integration/vllm/kv_cache_group_edits.py`、`lmcache/v1/kv_layer_groups.py`、`lmcache/v1/distributed/api.py`、`lmcache/v1/multiprocess/{object_group_transfer.py, modules/lmcache_driven_transfer.py}`、`docs/source/mp/lazy_offload.rst`。
- **寫入時做了什麼決定**：**幾乎沒有選擇——每個 chunk、每個層群組都存**。
  - 文件：「它為每一個群組存取 KV cache」〔程式碼 LMCache@7d7ca47 docs/source/mp/hybrid_models.rst:9–11〕。
  - Mamba／GDN：把 `[conv_state, ssm_state]` 重新解讀成一個不透明的 page，當成「視窗＝block 大小的滑動視窗層」，「只有最後一個命中的 block 會被用到」〔程式碼 lmcache/integration/vllm/kv_cache_group_edits.py:237–250〕；需要 `--separate-object-groups`，讓遞迴狀態和全注意力層分開存〔程式碼 hybrid_models.rst:120–143〕。
  - **存哪些位置**：vLLM 用 `--mamba-cache-mode align` 時只在每個 scheduler step 結束、block 邊界上快照；`--max-num-batched-tokens` 在 [N, 2N) 時「每個 block 邊界都會被快照」，≥2N 時只存每個 step 的最後一個 block〔程式碼 hybrid_models.rst:243–251〕。也就是位置由 step 大小決定，不是由 LMCache 選〔判讀〕。
  - 滑動視窗層：只有「視窗小於一個 LMCache chunk」時（例 DSV4），每個 chunk 只存最後一段視窗；`enable_full_sw_kv` 可改成全存〔程式碼 lmcache/v1/kv_layer_groups.py:572–602〕。視窗大於 chunk 的模型（Gemma-3、gpt-oss）則每個 chunk 都全存〔判讀，依同段程式碼的條件 `sw_size_tokens >= tokens_per_chunk → 回傳整個 chunk`〕。
  - **明說還沒做**：設計文件把「sliding-window load-plan trimming」列為不在範圍內〔程式碼 docs/design/integration/vllm/hybrid-kv-cache-groups.md:30〕；`kv_cache_group_edits.py` 的開頭把「每群組的 store/load 遮罩（SWA／Mamba 只傳尾段）」列為延後項目，並指向 vLLM 的 Mooncake store connector（PR #42828）的 `store_mask`／`load_mask` 當參考〔程式碼 kv_cache_group_edits.py:13–22〕。
- **讀取端（不是寫入時）**：命中規則按群組的視窗算——`num_chunks_in_sw[g]` 是群組 g 命中所需的尾端 chunk 數，-1 表示整個前綴〔程式碼 lmcache/v1/distributed/api.py:362–369〕；讀回時跳過視窗外的物件〔程式碼 object_group_transfer.py:360–369；modules/lmcache_driven_transfer.py:998–1011〕。
- **用什麼資訊做決定？寫完後還在不在？（N1）**：寫入端只用模型結構與 vLLM 給的 block 清單，沒有用重疊分布或命中次數。Mamba 中間狀態是 N1（vLLM align 模式下只在 step 邊界寫過一次）〔判讀〕。
- **有沒有和延後版、寫穿版、背景版比較？**
  - 有一個通用的延後版：`lazy offload`，預設策略 `EVICTION_AWARE`——「當持有資料的 GPU block 快被 vLLM 淘汰時才釋出 store」，FIFO 為另一選項〔程式碼 docs/source/mp/lazy_offload.rst:4–16〕。它和混合模型（Mamba 狀態被 vLLM 回收）能不能一起用，文件沒寫〔未查證〕。
  - 沒有「全存 vs 只存部分狀態」的比較。正確性驗證只做分數級：「GDN 後端不支援 batch-invariant 模式，所以不是 bit-exact」〔程式碼 hybrid_models.rst:273–276〕。
- **硬體、各層頻寬、模型**：已驗證的混合模型：Gemma 3／4、gpt-oss（SWA＋全注意力）；Qwen3.5／3.6／3.8、Qwen3.8-Flash-Next、Kimi-Linear、Kimi K3（GDN／KDA＋全注意力）等〔程式碼 hybrid_models.rst:22–64〕。unified block size N：Qwen3.6-27B 784、Qwen3.5-0.8B 544、Kimi-Linear 944〔程式碼 hybrid_models.rst:207–228〕。頻寬：未查證。
- **和 D5(b)/(c) 的關係**
  - **(b)**：LMCache 會把 Mamba 狀態存到 CPU（L1）與 L2 storage，但**寫入時沒有選擇**：存哪些位置完全跟著 vLLM 的 step 大小走。這是 H7「寫穿版」的現成實作——每個 block 邊界都存〔判讀〕。如果 vLLM upstream 的 retention interval 遮罩（見 D5_vLLMRetentionInterval）只作用在 OffloadingConnector，LMCache 這條路徑就是「全存」〔判讀；LMCache 是否讀那個遮罩：未查證〕。
  - **(c)**：LMCache 做了**讀取時**的分層感知（只讀視窗內），**寫入時**只對「視窗 < chunk」的模型裁切；Gemma-3／gpt-oss 這類視窗 ≥ chunk 的模型，視窗外的 KV 照存〔判讀〕。依 D5 判準 (c)：LMCache 屬於「只有部分做了」。
- **證據等級**：程式碼與文件〔程式碼 repo@commit file:line〕；Gemma／gpt-oss 寫入端全存是由程式碼條件推得〔判讀〕，沒有實跑。
