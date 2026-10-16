# P07｜大掃描要參考的前作：格式、佔比、放法、系統 SOTA、反量化加速

> 回到 [02](../02_papers.md)、[09 大掃描計劃](../09_sweep_plan.md)。寫於 2026-10-15（UTC 22:10–22:50）。
> **證據等級**：這張卡是**網路搜尋＋少數頁面直接讀**，不是 P01–P06 那種逐頁讀全文。
> - **B**：讀了原始頁面（官方部落格全文）。
> - **C**：只讀摘要，或搜尋摘要轉述。數字照轉述寫，**沒有回原文核對**。
> - 寫進論文之前，C 級的每一條都要回原文核對，照 P01–P06 的規格。

**一句話**：
- **格式**：4 bit 這一階最常見的是 NVFP4、MXFP4（硬體格式），加上旋轉（Hadamard）和 K／V 不同精度。2–3 bit 要靠旋轉或向量量化才不崩。
- **佔比**：「重要 token 給高精度」的比例，多數論文只掃一兩個點。**我們沒找到在 FP8～FP4 之間、以 5% 為一格掃佔比的研究**（只能說「沒找到」，不能說「沒有」）。
- **系統**：主流 serving 系統的分層快取（HiCache、LMCache、Dynamo）都**沒有依位置或依 block 混精度**。NVFP4 KV 已經進了 SGLang 和 TensorRT-LLM，但都是整段同一格式。
- **反量化**：多篇論文都指出反量化是低位元 KV 的瓶頸，也都有加速做法。**我們自己量到的反量化，只用了 HBM 頻寬的一成不到**（09 §6）。

---

## 1. 格式：4 bit 這一階有哪些選擇

| 格式／方法 | 怎麼存 | 別人量到的（照轉述） | 等級 | 出處 |
|:--|:--|:--|:--|:--|
| **NVFP4 KV**（NVIDIA） | E2M1；每 16 個值一個 E4M3 scale；另有 FP32 per-tensor scale | Qwen3-Coder-480B：Ruler 64K FP16 約 95.6、FP8 約 95.5、NVFP4 約 94.6。宣稱比 BF16／FP8「掉不到 1%」。**實作上 attention 前先反量化成 FP8** | B | [NVIDIA 部落格](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/) |
| **NVFP4 vs MXFP4** | MXFP4：每 32 個值一個 2 的次方 scale（E8M0） | Llama 3.3 70B MMLU：FP8 約 82.5、NVFP4 約 81.9、**MXFP4 約 77.8**。文字寫「NVFP4 比 MXFP4 高 5%」，圖上約 4.1 點 | B | 同上 |
| **NVFP4 KV in SGLang** | 同上；global scale 設 1.0（沒用到 FP32 那一層）；prefill 寫入時從 FP8 再量成 NVFP4，chunked prefill 時解回共用的 FP8 工作區 | 和 FP8 KV 比（**沒和 BF16 比**）：Qwen3.8-27B GSM8K −0.31、GPQA −1.01、SWE-bench Verified 76.20% vs 77.80%。**實驗性**；只在 Blackwell | B | [LMSYS 部落格 2026-09-16](https://www.lmsys.org/blog/2026-09-16-nvfp4-kv-cache) |
| **Hadamard 旋轉＋INT4**（QuaRot） | 先旋轉打散離群值，再 4 bit | LLaMA2-70B 4 bit（權重＋激活＋KV）WikiText-2 PPL 最多 +0.47 | C | [arXiv 2404.00456](https://arxiv.org/html/2404.00456v1) |
| **RotateKV**（2025） | 2 bit；Walsh-Hadamard＋channel 重排；RoPE 前旋轉；保護 sink | LLaMA-2-13B 2 bit PPL 退化 < 0.3 | C | [arXiv 2501.16383](https://alphaxiv.org/abs/2501.16383) |
| 區塊對角 Hadamard（2026） | INT4＋區塊對角旋轉，融合 kernel | 宣稱接近無損、端到端沒有可量到的額外開銷 | C | arXiv 2604.19157（只見轉述） |
| **KVTuner**（ICML 2025） | **逐層選 K、V 的精度對**（K 比 V 敏感） | Llama-3.1-8B 3.25 bit、Qwen2.5-7B 4.0 bit「接近無損」（另一版本寫 5.0 bit，要核原文） | C | [arXiv 2502.04420](https://arxiv.org/pdf/2502.04420v4) |
| **TurboQuant**（Google） | PolarQuant（極座標，省掉逐組 scale）＋1 bit QJL 修正殘差 | 宣稱 3 bit／值、LongBench 不輸 KIVI；**只見新聞轉述** | C | [新聞轉述](https://helpnetsecurity.com/2026/03/25/google-turboquant-ai-model-compression) |
| KIVI 2-bit 的崩潰 | per-channel K／per-token V | MiniKV：INT8／INT4 各種快取預算都穩，**INT2 會崩**；HF 部落格：int2 在 SAMSum、TriviaQA 掉很多 | C | [MiniKV](https://arxiv.org/pdf/2411.18077)、[HF 部落格](https://huggingface.co/blog/kv-cache-quantization) |

**對大掃描的意思**：
1. 格式軸加入 NVFP4、MXFP4、INT4＋Hadamard、INT2（KIVI、旋轉）、K8V4、K4V2（[m10_codec_x.py](../../../code/m10_codec_x.py)）。
2. NVFP4 和 MXFP4 的差（scale 的粒度和型別）在我們的模型上量得到。NVIDIA 只報一個模型、一個任務。
3. **旋轉對 Qwen 可能沒用**：Qwen 的 K 是「固定的大偏移」，旋轉會把偏移攤到每個 channel。我們的合成資料自測已經看到這個現象（`m10_codec_x` 自測的 `note_rotation_constant_offset`）。這是可以量的預測。
4. TurboQuant／CommVQ 這種向量量化沒做。理由：要碼本或特殊 kernel，而且我們只讀到新聞轉述。

## 2. 佔比與「哪些 token 給高精度」

| 前作 | 怎麼選高精度 | 佔比怎麼掃 | 等級 | 出處 |
|:--|:--|:--|:--|:--|
| ZipCache（NeurIPS 2024） | 正規化的 attention 分數 → 重要的 4 bit、其他 2 bit | 摘要沒寫掃佔比；轉述的後續作（ZipVL）有一張門檻掃描圖 | C | [arXiv 2405.14256](https://www.arxiv.org/pdf/2405.14256)、[ZipVL](https://arxiv.org/pdf/2410.08584) |
| **More Tokens, Lower Precision**（2024–25） | 同樣預算下「4 倍 token、4 bit」勝過「1 倍 token、16 bit」 | 掃預算；說對不同的剪枝法、量化法、模型大小都成立 | C | [arXiv 2412.12706](https://arxiv.org/abs/2412.12706) |
| LogQuant（ICLR 2025 workshop） | 「log 分布的過濾」選要保護的 token；2 bit | 同樣壓縮比下 Math／Code 準確率 +40–200%（轉述）。**選法摘要沒寫清楚** | C | [arXiv 2503.19950](https://arxiv.org/abs/2503.19950) |
| KVmix（2025） | 最近的關鍵 token 全精度、舊的壓；逐層位元由梯度決定 | — | C | [arXiv 2506.08018](https://arxiv.org/html/2506.08018v3) |
| Block-GTQ（2026-06） | 依 **RoPE 頻率區塊**（不是 token 位置）配位元 | — | C | [arXiv 2606.24033](https://huggingface.co/papers/2606.24033.md) |
| 已在 P03／P05 的 | SKVQ、KIVI、KVQuant、PM-KVQ、DiffKV、SemKV、PatchKV、MoQAE、Cocktail | 見 P03 §3.2、P05 | — | — |

**對大掃描的意思**：
1. **我們的佔比軸是空白處**：
   - 別人多半只比「一種重要度選法 vs 均勻」。
   - 很少把「同位元組、隨機選」當基準。SemKV 是例外，它在 Mistral 上隨機明顯比較差（P03）。
   - 也很少掃佔比的整條曲線。
2. **「More Tokens, Lower Precision」提醒還有另一條軸：丟掉（0 bit）vs 量化**。我們的設定裡，被丟掉的 block 要重算。在 Cake 下這等於「前段不用存」，不用新的格式就能量到（09 §8 列為延伸）。
3. 放法加入：
   - ATT：觀察窗的 attention，SnapKV／ZipCache 式。
   - ERR：寫入當下就算得出來的重建誤差，SemKV、PatchKV 式。
   - AXE：attention × 誤差，PatchKV 的風險分數。
   - ORC：作弊上限。
4. LogQuant 的選法摘要沒寫清楚，所以沒放進去，免得照我們的猜測實作、卻掛它的名字。

## 3. 系統層 SOTA（分層 KV 快取與壓縮）

| 系統 | 分層 | 有沒有壓縮／混精度 | 等級 | 出處 |
|:--|:--|:--|:--|:--|
| **SGLang HiCache** | GPU（L1）→ host（L2）→ 分散式儲存（L3：Mooncake、3FS、NIXL、AIBrix、本機檔案） | 搜尋結果裡沒看到壓縮；宣稱吞吐最多 6 倍、TTFT 少 80% | C | [LMSYS 2025-09-10](https://lmsys.org/blog/2025-09-10-sglang-hicache)、[Dynamo 文件](https://docs.nvidia.com/dynamo/v1.0.0/integrations/sg-lang-hi-cache) |
| SGLang NVFP4 KV | GPU 內 | 整段 NVFP4（實驗性） | B | §1 |
| TensorRT-LLM／Model Optimizer | GPU 內 | FP8、NVFP4 KV（整段同格式） | B | §1 |
| LMCache | CPU、磁碟、遠端；CacheGen codec | 現行版只剩固定 codec（P02）；技術報告不評估壓縮（P06） | A（P02、P06） | — |
| vLLM | KV offloading connector；`--kv-cache-dtype` | 只有逐層跳過量化，沒有逐 block（P04） | A（P04） | — |
| **Tutti**（2026） | SSD 上的 KV | 重點是存取形狀（128K token、64 層要讀約 25.6 萬個小物件）；新版引擎上 SSD 還原可能輸給重算 | C | [arXiv 2605.03375](https://arxiv.org/pdf/2605.03375) |
| **DirectKV**（OSDI'26） | GH200 上 GPU 直接讀 CPU 的 KV（NVLink-C2C） | 宣稱傳輸量少 50%、端到端 1.2 倍 | C | [papers.cool](https://papers.cool/venue/luo@osdi26@USENIX) |
| 方法論：Rethinking KV compression（MLSys 2025） | — | 壓縮省了記憶體，吞吐卻沒跟著提升（實作沒針對 serving 最佳化）；壓縮會讓輸出變長；要看逐題而不是平均 | C（P06 已部分讀） | [arXiv 2503.24000](https://arxiv.org/pdf/2503.24000) |
| 評測工具：NVIDIA **KVPress** | 15 種以上的壓縮法（丟 token 為主）；RULER／LongBench；Llama-3.1-8B 排行榜 | 主要是丟 token，不是量化 | C | [排行榜](https://nvidia-kvpress-leaderboard.hf.space) |

**對大掃描的意思**：
1. 「每個 block 自己選精度＋分層＋重算」在主流系統裡**都沒有**。所以如果位置策略有用，系統面是空白；如果沒用，「整段同格式」就是對的設計。兩種結果都有話可說。
2. Rethinking 提醒兩件事：**要報逐題**，不能只報平均；**省記憶體不等於快**。所以大掃描每一列都記「位元組」和「推算的還原時間」，複賽再實測。
3. 用 KVPress 的 RULER 設定當外部效度是選項，但它的方法多半是丟 token，不在這輪範圍。

## 4. 反量化的開銷和加速（使用者 10/15 要求）

| 前作／套件 | 講什麼 | 等級 | 出處 |
|:--|:--|:--|:--|
| **QServe**（MLSys 2025） | INT4 的 KV 反量化每個值要 5 個 ALU 運算。在 A100 上，光是反量化就讓融合 attention 變成「算不完」（compute-bound），8→4 bit 反而慢 1.2 倍（L40S 快 1.7 倍）。解法：FP32 運算改 FP16、位元技巧、預先載入 scale／zero | C | [arXiv 2405.04532](https://arxiv.org/pdf/2405.04532) |
| **BitDecoding**（2025） | 低位元 KV 的反量化改到 Tensor Core 上做；對 FP16 FlashDecoding-v2 宣稱平均 7.5 倍；Blackwell NVFP4 最多 8.6 倍；比 QServe 快最多 4.3 倍（作者自報） | C | [arXiv 2503.18773](https://arxiv.org/html/2503.18773v3) |
| vLLM FP8 Marlin | 用位元運算一次把 4 個 FP8 轉成 FP16／BF16；記憶體受限時最多 2 倍 | C | [vLLM PR 5975](https://github.com/vllm-project/vllm/pull/5975) |
| FLUTE（2024） | 查表式反量化 kernel；離線重排權重、複製查表避開 shared memory 衝突；小 batch 快 2–4 倍 | C | [arXiv 2407.10960](https://arxiv.org/html/2407.10960v1) |
| **Petit**（AMD，MI250／MI300） | FP4→BF16：8 個 FP4 原本要 31 個指令，用 AMD 的位元組轉換指令降到 15 個 | C | [ROCm 部落格](https://rocm.blogs.amd.com/artificial-intelligence/fp4-mixed-precision/README.html) |
| **ShadowServe**（2025） | CacheGen 在 GPU 上解壓，**和 decode 同時跑會互相拖慢**（L40S、30K）；改把解壓放到 SmartNIC | C | [arXiv 2509.16857](https://arxiv.org/pdf/2509.16857) |
| KVFetcher（2026） | CacheGen 的 CUDA core 解壓太貴 → 改用 GPU 內建的影片解碼器（NVDEC） | C | [arXiv 2602.09725](https://arxiv.org/pdf/2602.09725) |
| PackKV（2025） | 解壓直接融合進矩陣－向量 kernel，不寫回 global memory | C | [arXiv 2512.24449](https://arxiv.org/pdf/2512.24449v1) |
| CacheGen | GPU 解碼和傳輸做成管線，所以端到端影響小 | A（P02） | — |
| AMD AITER／vLLM ROCm | 有 FP8 KV 的 paged attention（HIP 版、ASM 版兩條路）。**反量化在 kernel 裡怎麼做，文件沒寫** | C | [ROCm vLLM 調校文件](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html) |
| MI300X 的 FP8 方言 | MI300X 硬體是 **fnuz** 版 FP8（偏差差 1，同一個位元組讀錯方言會差 2 倍）；MI325／MI350／MI355X 才是 OCP 標準。我們的 codec 存 OCP e4m3fn（為了和 H200 相同，P04），所以在 MI300X 上只能軟體轉換 | C | [doubleword 部落格](https://blog.doubleword.ai/deepseek-v4-flash-mi300x) |

**和我們量到的對照**（〔實測〕見 08 §2.1、09 §6）：
- **「解壓和計算搶 GPU」我們也量到了**：Qwen 在真實 CPU 層，FP8 反量化 GPU 閒時 6.8 ms、忙時 26.0 ms（64 個 block）。這和 ShadowServe 說的是同一件事。
- **我們的反量化只用到 HBM 頻寬的不到一成**〔算術：Llama 一個 block 讀 32 MiB、寫 64 MiB，0.23 ms → 約 0.4 TB/s；MI300X 約 5.3 TB/s〕。原因是 PyTorch 拼的寫法把同一份資料來回搬 4–5 趟。
- **第一步的優化不需要 tensor core 這種重武器**：我們的反量化在 attention **之外**（解成 BF16 後走和 E0 同一條 attention，07 的要求）。融合成「讀一次、寫一次」的 kernel 就夠了。已寫好 `code/m10_dequant_fast.py`（Triton，查表轉 FP8），和原寫法**逐位元相同**；速度在 09 §6 量。
- 下一步才是 QServe／BitDecoding 那種「融合進 attention」。但那會換 attention 路徑，違反 07 的「同一條 attention」，只能當系統面的延伸。

## 5. 沒查到的（照實列）

- 沒找到「同時比 FP8、INT8、INT4、INT2、FP4 多種 KV 格式、在 RULER＋LongBench 上」的系統性研究。搜尋到最接近的是一篇 workload-aware benchmark（KIVI4 最穩、壓縮比不能預測端到端效能），**原文沒讀到**。
- CommVQ、LiquidGEMM、KVServe 的解壓時間拆解：搜尋沒有結果。
- MI300X 上 `v_cvt_pk_f32_fp8` 這類 FP8 轉換指令的細節：沒查到。要看 CDNA3 ISA 文件。
- 上面所有 C 級的數字，都沒有回原文逐頁核對。
