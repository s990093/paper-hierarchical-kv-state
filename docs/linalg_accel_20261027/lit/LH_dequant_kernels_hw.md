# LH：還原時「反量化／解壓／低秩重建」的 kernel、指令與硬體（H200、MI300X、Zen 4 CPU）

> 文獻 agent H，2026-10-27。規則見 `RULES.md`。這台 Mac 上沒有跑任何實驗；表裡的「實測」都是原論文或 repo 已有的 run，〔算術〕都寫出公式與代入值。
> 分工：數學面（把基底吸收進 query、和 RoPE 交換、scale 吸收、LUT／ADC）是 agent G；本檔只管 **kernel／指令／硬體**。硬體峰值不重抄，見 `LF_hw_isa.md` §1（F9、F10、F16）。
> 證據等級：〔全文〕〔部分〕〔摘要〕〔程式碼〕〔文件〕〔二手〕〔算術〕〔判讀〕〔未查證〕。**廠商部落格的數字標〔文件〕，不是獨立量測。**
> 下載的 PDF 在 `<SCRATCH>/litH/`，頁碼一律是 PDF 頁（form feed 計數）。

---

## §0 一頁結論（白話）

**一句話**：在我們的還原裡，「把 INT4／FP8 轉回 BF16」已經快到不用再加速（H200 每塊 0.03–0.09 ms，只佔搬運時間的 1/5 以下，而且可以和下一塊的搬運重疊）；**真正會出事的是低秩重建，而它出事的原因不是算太多，是 GPU 忙時「排不到 SM／CU」**。所以硬體面該做的不是更聰明的位元技巧，而是 **(1) 把重建做成一個（或很少個）kernel、(2) 給它自己的 SM／CU 份額，(3) 永遠在 GPU 上解，不要在 CPU 上先解再搬。**

分五點：

1. **反量化已經接近 HBM 頻寬的上限，再快也省不到什麼。**
   - repo 量過：MI300X 上 INT8→BF16 一個 64 MiB 塊 0.031 ms（約 3.3 TB/s，峰值的 62%）；H200 上融合 kernel／torch.compile 0.03–0.09 ms（1.3–3.4 TB/s）。
   - 用 PCIe 搬同一塊 INT4 要 0.365 ms〔算術〕。所以反量化只佔搬運的 8–20%（MI300X X3 為 12%），而且能和下一塊的搬運重疊。
   - 論文裡的位元技巧（FasterTransformer 的「魔術數字＋LOP3」、Marlin、QServe 的「先乘後減」）是為了**在注意力或 GEMM 的內迴圈裡**反量化：那裡 CUDA core 會被算滿（QServe 在 A100 上量到 KV4 注意力反而比 KV8 慢 1.1–1.2 倍）。**我們的還原是獨立的一趟「讀小、寫大」，被頻寬卡住，這些技巧幫不上大忙。**
2. **在 tensor core 上直接用低精度算 QKᵀ：H200 可以用 FP8 和 INT8，不能用 INT4；MI300X 硬體有 FP8（FNUZ）和 INT8，但沒找到量過的注意力 kernel。**
   - Hopper 的 `wgmma` 只吃 f16／bf16／tf32／e4m3／e5m2／s8／u8，**沒有 s4**（PTX ISA 9.4）。SageAttention2 也寫 Hopper「缺少原生 INT4 tensor core」。
   - 所以 INT4 KV 在 H200 上一定要先在暫存器裡轉成 FP16／BF16。BitDecoding 的做法是：LOP3 轉換 → 用 STSM 寫進共享記憶體 → 再給 `wgmma`。
   - FP8 注意力（FA3）在 H100 上有實測，但長 context 有精度陷阱：vLLM 量到 128K 撈針從 91% 掉到 13%，用「兩層累加」修回 89%。
3. **低秩重建（係數 × 基底）在秩夠大時是算力型的。**
   - 每輸出 1 byte 要做約 r 次運算（r＝聯合秩）〔算術〕。Llama 的 xKV k22（r＝704）遠高於 H200 的平衡點 189 FLOP／byte；Qwen k11（r＝176）在邊界；每 head 的低秩（r＝16–64）是頻寬型。
   - repo 在 MI300X 上量到重建只用到 BF16 峰值的 13–28%〔算術，由實測時間換算〕。
   - **GPU 忙時，重建變慢的倍數和 FLOPs 無關**：r＝16 的每 head 低秩，FLOPs 只有 xKV f4 的 1/64，卻慢 5.3 倍（xKV f4 是 3.1 倍）。這和 repo 先前的推論一致（11 §3d）：瓶頸是 kernel 和背景搶 CU、排隊，不是 tensor core 被算滿。
4. **隔離的工具：MPS 量過有效，stream priority 沒效，Green Context／CU mask 只保證「分 SM」、不分頻寬。**
   - H200（repo B0）：MPS 限 87% 時，cprefill 背景下「INT4 全載入」比同 process stream 快 4.28 倍；stream_hi（高優先 stream）和普通 stream 幾乎一樣。
   - MI300X（repo 11 §3d）：留 3/4 CU 只對 Cake 有用，背景慢 12–20%；**把所有塊搬完再一次拆（一個 kernel）**最有效，忙時拆的時間從 9.7–23.4 ms 降到 1.4–4.3 ms。
   - 反量化搬不進 copy engine：Hopper／CDNA3 的 DMA 不做資料轉換；只有 Blackwell 的解壓引擎能邊傳邊解，而且只吃無損格式（LF F13）。
5. **CPU 端先解再過 PCIe：在我們的平台上永遠輸。**
   - INT4→BF16 讓 PCIe 位元組變 3.2 倍，DRAM 流量變 7–10 倍。
   - 要贏，GPU 反量化一塊得慢到 ≥0.80 ms，是 H200 實測值的 9–29 倍〔複核 2026-10-27：原寫「12–16 倍」，和 §4.5 自己的 0.80÷(0.028–0.087)＝9–29 倍不一致，改為 9–29 倍〕；目前量過最差的爭用只慢 1.3–3.8 倍。
   - CPU 的 SIMD 本身不是瓶頸：Zen 4 的 `VPERMB`、`VPSHUFB` 每週期 1 條，粗估 16 核轉換速度遠超過 DRAM〔算術，未實測〕。卡住的是 DRAM 頻寬。

**查不到的**：MI300X 上 FP8／INT8 注意力 kernel（AITER／CK）的實測 TFLOPS；Zen 4／SPR 上 int4→bf16 解包的實測 GB/s；skinny GEMM（K＝r≈500–1000）在 H200／MI300X 上的官方或論文效率；Green Context 在 H200 上對還原的實測隔離度。

---

## §1 總表

「量到的速度」只抄原始來源；「對我們的還原有沒有用」是〔判讀〕。還原＝KV 從 CPU／SSD 搬回 GPU 之後、注意力之前的那一步。

| 平台 | 技巧／指令 | 做什麼 | 量到的速度（出處、設定） | 條件 | 對我們的還原有沒有用 |
|:--|:--|:--|:--|:--|:--|
| NVIDIA（sm_80+） | 魔術數字＋`lop3`（Kim 2022；FT／TRT-LLM `interleaved_numeric_conversion.h`） | INT4／INT8 → FP16（`0x6400`）或 BF16（`0x4300`）：把整數 OR 進尾數再減掉偏移，兩個值一起轉 | V100：融合 GEMM＋反量化，INT4 比 FP16 平均 1.56×（MoE GEMM，A＝m×1024、B＝1024×4096）〔全文〕Kim Table 1 | 權重要預先交錯排列（[e0,e2,e4,e6,e1,e3,e5,e7]） | 低：我們的獨立反量化已經被頻寬卡住；放進注意力 kernel 才有用 |
| NVIDIA Ampere（A10、3090、A100） | Marlin | FP16×INT4 GEMM；權重預先重排，`lop3`＋減法轉換，直接排成 tensor core 的暫存器版面 | A10：batch 16–32 以內接近理想 3.87×〔全文〕Marlin §6 p.9；A100 加速較小 | 用 `mma.sync`，不用 Hopper 的 `wgmma` | 低（為權重設計；H200 上要改用 Machete） |
| NVIDIA Hopper | Machete（vLLM） | 混合精度 GEMM，用 TMA＋`wgmma`；算 Yᵀ＝Wᵀ·Xᵀ，讓轉好的權重當 A 留在暫存器 | 只有圖；文字說 batch ≥128 時和 FP16 相當；端到端 Llama-3.1-70B／1×H100 吞吐快 29–32%〔文件，vendor blog〕；**只用 mma 在 Hopper 上約損失 37% 峰值**〔文件〕 | Hopper 限定 | 間接：說明 Ampere 式 kernel（含 repo 的 Triton）在 H200 上拿不到峰值 |
| NVIDIA A100／L40S | QServe KV4 注意力 | decode 注意力裡反量化 INT4 KV；FP32 運算改 FP16、用 Kim 的位元技巧把反量化降到每元素 2 個運算 | 最佳化後比 TRT-LLM KV8 快 1.5×（A100）；naive KV4 在 L40S 快 1.7×、在 A100 反而慢 1.1–1.2×〔全文〕QServe §5.3 p.10 | 融合在 attention 內 | 教訓：反量化放進 kernel 內迴圈會讓 CUDA core 變成瓶頸（A100 FP32 平衡點只有 9.8 ops／B） |
| NVIDIA Ampere／Ada／Hopper／Blackwell | BitDecoding | 低位元 KV decode：`ldmatrix` 載入 packed 資料，`lop3` 轉換，再進 tensor core；Hopper 版用 STSM 寫回共享記憶體再給 `wgmma` | H100：比 FlashDecoding-v2 最高 8.0×（v3 實作）；Blackwell NVFP4 最高 8.6×；比 QServe 最高 4.3×〔部分〕§V-B、§VI | 基準是 FP16 FD-v2，不是 FA3 | 中：H200 上 INT4 KV「不先寫回 BF16 就算注意力」的現成作法；換 attention 路徑違反 repo 07「同一條 attention」 |
| NVIDIA A100 | KIVI 的 kernel | 反量化融合進矩陣乘（CUDA），分組量化（Triton） | 量測只有端到端（A100）〔部分〕§4、p.6、p.8 | — | 低 |
| NVIDIA A6000／A100 | FLUTE（LUT） | 查表式反量化 GEMM；離線重排、複製查表避開共享記憶體衝突 | batch <32、group 128 時比既有 kernel 快 2–4×〔摘要＋圖〕 | 非均勻碼本 | 只對碼本式壓縮有用（G 的範圍） |
| 未查證 GPU | LUT-GEMM（ICLR'24） | BCQ 權重用查表做 GEMV | 對 OPTQ 2.1×（單 GPU，型號這次沒讀到）〔摘要〕 | 權重 | 無 |
| NVIDIA／AMD MI250 | Ladder／BitBLAS（OSDI'24） | 編譯器：自訂低位元型別的張量版面轉換 | 不支援的型別最高 14.6×（A100、V100、A6000、MI250）〔部分〕p.3 | — | 低；有 AMD 後端是少見的優點 |
| **H200** | `wgmma` FP8（e4m3／e5m2）、INT8（s8／u8） | QKᵀ、PV 直接用 8 位元算 | FA3 H100：FP8 約 1.2 PFLOPS（LF F10）；SageAttention2-8b（INT8 QK＋FP8 PV）速度＝FA3-fp8（H100 2.61× vs 2.63×、H20 3.12× vs 3.06×，對 FA2）〔全文〕Table 9 | **沒有 s4**（PTX ISA 9.4）；FP8 累加只有約 22 位元（FP22）〔全文〕SageAttn2 §3.4 | 「新問題在縮圖上 prefill」可用（LF §4.2）；長 context 精度要驗 |
| H200 | vLLM FP8 KV（FA3） | QKᵀ、PV 全部在 FP8 | H100，Llama-3.1-8B，單請求：decode ITL 斜率降到 BF16 的 54%，打平點 7,010 token；128K 撈針 91%→13%，兩層累加後 89%〔文件，vLLM blog〕 | per-tensor scale＝1（未校準） | 有：若縮圖存 FP8，可不經 BF16 直接用 |
| H200 | repo 融合反量化（`code/m10_dequant_fast.py`；k1／k3／k4） | FP8／INT8／INT4／KIVI → BF16，讀一次寫一次，寫進 KV 視圖 | Llama 一塊 64 MiB：k2（torch.compile）閒 0.060–0.066、忙 0.028–0.030 ms（int4_kivi 除外：k2 閒 0.114、忙 0.085）；k1 閒 0.071–0.087、忙 0.049–0.078 ms〔複核 2026-10-27：原寫 k2「閒 0.060–0.066、忙 0.028–0.030」未註明不含 int4_kivi，k1 閒下界原寫 0.073（fp8_head 是 0.0705）；照 CSV 補正〕；k0（PyTorch）0.34–0.65 ms〔實測 `results/m10_sweep_h200/dqsweep/A/kernels_llama.csv`〕 | 逐位元和 PyTorch 相同 | **就是我們用的那條**；已接近頻寬上限 |
| MI300X | repo F3 反量化 | 同上（Triton／torch.compile） | INT8 0.031、FP8 0.026、INT4 0.027 ms／64 MiB（cold）≈3.3 TB/s，峰值 62%〔實測 `F3_dequant_fp8.md` §4.1〕 | — | 同上 |
| MI300X | MFMA FP8（FNUZ）、INT8 | 8 位元矩陣乘 | 峰值 FP8／INT8 2,614.9（稠密）〔文件〕；注意力實測**沒找到** | **INT4 不在矩陣型別表**（CDNA3 白皮書 p.7、p.24） | vLLM ROCM_ATTN 讀 FP8 cache 時在 kernel 內乘 scale 轉回（F3 §4.4，程式碼），**不是 FP8 運算** |
| MI300X | SageAttention | — | 只有 Triton v1 理論上可跑，作者未測（GitHub issue #20、#125）〔二手〕 | — | 沒有可用的 port |
| A100 | ShadowKV 重建 | 只搬 V；pre-RoPE K 用低秩（rank 160）在 GPU 重建＋RoPE，融合 kernel，和 V 的搬運用不同 stream 重疊 | Llama-3-8B-1M 每層 decode：重建 K 1.25–1.51 ms，被同時的「搬 V」1.66–1.84 ms 蓋過〔全文〕Table 13 p.18 | 稀疏預算 1.56% | 有：「重建和搬運重疊」是現成的排程；repo X3 是相加、沒重疊（§3.4） |
| RTX 4090 | Palu 融合重建 | 在注意力 kernel 裡從低秩潛在向量即時重建 K（RoPE 模型） | 64K 時注意力模組 1.89×；V 再加 4-bit 量化 2.91×（50% 壓縮）〔全文〕§4 p.9 | V 的基底併進 W_o（屬 G 的範圍） | 中：要換 attention 路徑 |
| A100 | xKV 重建 | 跨層共用基底；選擇性重建＋和 V 搬運重疊（照 ShadowKV） | 只有圖（Fig. 5）〔部分〕§5 | — | repo 已在用它的格式 |
| H200 | MPS（`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`） | 限背景的 SM 比例 | repo B0：cprefill 背景、mps87：INT4 全載入比 stream 快 4.28×，背景吞吐 0.98（GB0-SHARE）〔實測 `results/s3_h200/b0/b0_report.json`〕 | 跨 process；要另載一份權重 | **有**，已量過 |
| H200 | Green Context | 把 SM 切成子集給不同 stream | CC 9.0 對齊 8 個 SM；**不保證同時執行或前進**；只切 SM，不切頻寬〔文件〕CUDA Driver API | 同 process | 沒有 H200 的還原實測；ShadowServe 在 L40S 上評價差（repo L5c） |
| H200 | stream priority | 高優先 stream | repo B0／G-M08：lr_recon 在 stream 0.628、stream_hi 0.637（閒時的比例）〔實測〕 | — | **沒效** |
| MI300X | CU mask（`hipExtStreamCreateWithCUMask`） | stream 只用指定的 CU | API 文件沒說是硬性還是提示〔文件〕；repo：留 3/4 CU 只對 Cake 有用、背景慢 12–20%（11 §3d）〔實測〕 | — | 部分有用 |
| Zen 4（EPYC 9554／9684X） | AVX-512 `VPERMB`／`VPSHUFB`／`VPMOVZXBW`、`VCVTNE2PS2BF16` | 半位元組解包、轉 BF16 | 吞吐（週期／條）：VPERMB、VPSHUFB、VPMOVZXBW zmm 各 1.00；VCVTNE2PS2BF16 zmm 實測 1.35（文件 2.00）〔全文〕uops.info | — | CPU 先解再搬：**輸**（§4.5） |

---

## §2 卡片

### 2.1 位元技巧（大多是給權重用）

#### H1：Kim et al., *Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production*（SustaiNLP@EMNLP 2022；arXiv 2211.10017）
- **讀了什麼**：〔全文〕§3.4、Table 1、§4（PDF p.5–6），<https://arxiv.org/pdf/2211.10017>，查證 2026-10-27。
- **一句話**：把 MoE 專家權重量化成 INT8／INT4，在 CUTLASS GEMM 裡即時反量化。
- **硬體的關鍵一步**（§3.4.1–3.4.3，p.5）：
  - 原生的整數轉浮點（I2F）指令比預期慢，所以改成一串 ALU 和 FP16 指令。
  - 觀察 1：FP16 裡 1024 ≤ X < 2048 時，X−1024 的整數部分就直接在尾數位元裡。
  - 觀察 2：任何 0 ≤ Y < 1024，`0x6400 | Y` 就是 Y＋1024 的 FP16。
  - 所以一個 32 位元暫存器一次做兩個；INT8 減 [1152, 1152]（1024＋128，順便把無號轉回有號），INT4 減 [1032, 1032]。
  - INT4 要把 8 個值重排成 [e0,e2,e4,e6,e1,e3,e5,e7] 來減少邏輯運算。
  - 結果和原生 I2F 逐位元相同（§3.4.1）。
- **量到的數字**：單張 PCIe V100、CUDA 11.6（p.6）。Table 1：對 FP16 MoE GEMM 的吞吐，INT8 原生 I2F 幾何平均 1.26×、INT8 最佳化 I2F 1.35×、INT4 最佳化 I2F 1.56×。
- **論文沒提 `lop3`**：pdftotext 全文沒有這個字。`lop3` 出現在 TRT-LLM 的實作裡：`cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h`，FP16 的魔術數字 `0x64006400`（約 L227–252），BF16 版 `0x43004300`（約 L345–363），只在 `__CUDA_ARCH__ >= 800` 時啟用〔程式碼，main，查證 2026-10-27〕。
- **跟 Tiara 的關係**〔判讀〕：repo 的 `k4`（FP8 用位元運算拼 FP32、不查表，`code/m10_dequant_fast.py` `_dqm_kernel` 的 BITS 分支）就是同一類技巧。H200 實測 k4 比查表的 k1 還慢（Llama fp8_token 閒 0.121 vs 0.087 ms，`kernels_llama.csv`）。原因：我們的瓶頸是記憶體，不是 ALU。
- **與既有整理**：`docs/precision_20261013/cards/P07_sweep_formats_sys_dequant.md` §4 有 C 級轉述（vLLM FP8 Marlin 一次轉 4 個）。本卡補上原始出處與 TRT-LLM 的程式碼位置。

#### H2：Frantar et al., *MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models*（PPoPP'25；arXiv 2408.11743）
- **讀了什麼**：〔部分〕§4（反量化，p.6）、§6（p.9），<https://arxiv.org/pdf/2408.11743>，查證 2026-10-27。
- **一句話**：FP16×INT4 GEMM；batch 中等時仍拿到接近理想的「位元組少 4 倍 → 快 4 倍」。
- **硬體的關鍵一步**（p.6）：
  - 改自 Kim 的位元操作：用一條 `lop3` 取出 INT4 並 OR 上指數位元，再減一個常數（同時把減 8 併進去），兩個 INT4 一起轉。
  - 直接轉成下一個 tensor core 呼叫需要的暫存器版面。
- **量到的數字**（p.9）：
  - NVIDIA A10、72k×18k 矩陣、group 128：batch 到約 16–32 都接近 3.87×（4 位元＋scale 的 0.125 位元額外開銷）。
  - 3090 加速較大；A100 較小，原文歸因於 A100 頻寬和算力都高，固定開銷占比變大。
- **跟 Tiara 的關係**：Marlin 是 Ampere 的 `mma.sync` kernel。Hopper 的後繼是 Machete（H3）。

#### H3：Machete（Red Hat／Neural Magic 部落格，Lucas Wilkinson，2024-10-14）
- **讀了什麼**：〔文件，vendor blog，WebFetch 轉述〕<https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel>，查證 2026-10-27。
- **事實**：
  - Hopper 的 `wgmma` 只有 A 與 C 能從暫存器來，B 必須在共享記憶體。所以 Machete 算 Yᵀ＝Wᵀ·Xᵀ，讓轉好的權重當 A。BitDecoding 對 KV 也碰到同一個限制（H5）。
  - 部落格說「只用 mma 指令」在 Hopper 上會損失約 37% 的峰值算力〔文件〕。
  - 端到端：Llama-3.1-70B／1×H100、≥3 req/s，輸入吞吐快 29%、輸出快 32%（對其他混合精度 kernel 的幾何平均）。
  - kernel 層的數字只有圖。
- **跟 Tiara 的關係**〔判讀〕：repo 的重建 GEMM 走 `torch.matmul`（cuBLAS／hipBLASLt），在 H200 上應已用到 `wgmma`。但 repo 自寫的 Triton 反量化不碰 tensor core，這一點無關。

#### H4：Lin et al., *QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving*（MLSys'25；arXiv 2405.04532）
- **讀了什麼**：〔部分〕摘要、§5.3（p.10），<https://arxiv.org/pdf/2405.04532>，查證 2026-10-27。
- **一句話**：W4A8KV4 的系統協同設計；KV4 注意力的反量化要壓到不讓 CUDA core 變成瓶頸。
- **關鍵事實**（§5.3，p.10）：
  - naive 地反量化一個 INT4 KV 要 5 個 ALU 運算（mask、shift、轉型、乘、減）。
  - A100 FP32 CUDA core 的 roofline 轉折點只有 9.8 Ops／Byte，光反量化就把它吃滿。
  - 所以 naive KV4 注意力在 L40S 比 TRT-LLM KV8 快 1.7×，在 A100 反而慢 1.1–1.2×。
  - 解法：FP32 運算全改 FP16（算力上限加倍）；用 Kim 的位元技巧把反量化降到每元素 2 個運算；預先載入 scale／zero。
  - 結果：A100 上比 TRT-LLM KV8 快 1.5×。
  - 「先乘後減」（subtraction after multiplication）是為了能用 `vadd4`（一條 INT32 ALU 做 4 個 INT8 加法）實現暫存器層級平行（RLP），因為沒有 4 路 INT8 乘法指令（p.9–10）。這段是**權重** GEMM 的反量化，不是 KV。
- **跟 Tiara 的關係**〔算術＋判讀〕：
  - 我們的獨立反量化每個輸出 BF16（2 bytes 寫＋0.5–0.6 bytes 讀）做 2–5 個運算 → 約 0.8–2 ops／B。
  - H200 CUDA core 的平衡點約 67 TFLOPS（FP32，規格表）÷4.8 TB/s＝14；MI300X 是 163.4÷5.3＝31。
  - 兩者都遠高於 0.8–2 → **獨立反量化是頻寬型**，位元技巧省 ALU 不省時間。
  - QServe 的教訓只在「反量化放進 decode 注意力內迴圈」時成立。

#### H5：Du et al., *BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache*（arXiv 2503.18773）
- **讀了什麼**：〔部分〕摘要、§II–§V（p.1–9），<https://arxiv.org/pdf/2503.18773>，查證 2026-10-27。
- **一句話**：低位元 KV 的 decode 注意力，讓 tensor core 和 CUDA core 合作。
- **硬體的關鍵一步**：
  - packed KV 用 `ldmatrix` 載入，讓暫存器版面直接對上 `mma`；用 `lop3` 轉換（p.5）。
  - **Hopper**：`wgmma` 的 B 必須在共享記憶體，所以用 STSM 把反量化後的 FP16 寫回共享記憶體，再做 `wgmma_SS`；靠 `wgmma` 的非同步性把寫回和運算重疊（p.8）。
  - **Blackwell**：跳過 `lop3`，直接用 mxfp4／nvfp4 的低精度 mma（p.8）。
- **量到的數字**：
  - 對 FP16 FlashDecoding-v2：Blackwell 最高 8.6×、Hopper 8.0×、Ada 7.5×、Ampere 4.8×；比 QServe 最高 4.3×（p.2、§VI-A）。
  - H100：v2 實作最高 4.1×，v3 實作（`wgmma`＋非同步搬運）8.0×（p.9，Fig. 9）。
- **讀法注意**〔判讀〕：基準是 FD-v2（FP16），不是 Hopper 最佳化的 FA3。8× 裡有一部分來自「v2 → v3」，不全是低位元的功勞。
- **跟 Tiara 的關係**：H200 上「INT4 KV 不先還原成 BF16 KV 就做注意力」的現成作法。但這等於換 attention 路徑；repo 07 規定判定要用同一條 attention（P07 §4 已記）。
- **與既有整理**：P07 §4 有 C 級轉述（7.5×、8.6×、4.3×），與本卡一致；本卡補上 Hopper 的 STSM／`wgmma_SS` 細節。

### 2.2 低精度直接算注意力（不先反量化）

#### H6：Hopper 的 tensor core 型別（NVIDIA PTX ISA 9.4；H200 規格表）
- **讀了什麼**：〔文件〕PTX ISA 9.4 HTML（`wgmma.mma_async` 的「Matrix Data-types」與指令語法節；本次以文字檔行號定位，約 L34981–36560），<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>；H200 規格表 <https://www.nvidia.com/en-us/data-center/h200/>，查證 2026-10-27。
- **事實**：
  - `wgmma.mma_async` 的被乘數型別：`.f16`、`.bf16`、`.tf32`、`.e4m3`／`.e5m2`（dtype f16 或 f32）、整數 `.s8`／`.u8`（dtype s32）。**沒有 `.s4`／`.u4`**〔文件〕。
  - 舊的 `mma.sync`／`wmma` 的 `.u4`／`.s4` 需要 sm_75 以上（同文件）。但文件沒說 sm_90 上它是原生速度還是模擬〔未查證〕。
  - SageAttention2 原文兩處寫 Hopper「lack native INT4 tensor core support」〔全文 p.2、p.23〕。
  - H200 規格表只列 FP8／INT8 tensor core（3,958 含稀疏），**沒有 INT4**〔文件〕。
  - `.tf32` 的 `wgmma` 會截掉低 13 位元（同文件）。
- **跟 Tiara 的關係**：H200 上「INT4 KV 直接進 tensor core」不可能，一定要先在暫存器轉成 BF16／FP16（或 FP8）。

#### H7：Zhang et al., *SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization*（ICML'25；arXiv 2411.10958）
- **讀了什麼**：〔部分〕§1–§3.4（p.1–6）、附錄 Table 9、14、18、19（p.15、21、23），<https://arxiv.org/pdf/2411.10958>，查證 2026-10-27。
- **一句話**：Q、K 用 INT4（Hopper 上用 INT8），P̃、V 用 FP8 的注意力 kernel。
- **硬體的關鍵事實**：
  - Ada 與 Hopper 的 `mma(f32f8f8f32)` 累加器實際只有 FP22（1 符號、8 指數、13 尾數），低 10 位元被截掉（§3.4，p.6）。所以要「兩層累加」：先在 FP22 累加一小段，再加進真正的 FP32 暫存器。
  - Hopper 沒有原生 INT4 tensor core，所以提供 SageAttention2-8b（Q、K 用 INT8）（p.2、p.23）。
- **量到的數字**：
  - RTX 4090 峰值 481 TOPS（p.2）。
  - Table 9（對 FA2 的倍數）：H100 上 FA3 1.37、FA3-fp8 2.63、SageAttention2 2.61；H20 上 1.57、3.06、3.12（p.15）。
  - Table 14（H100、Llama-3-262k 8B、InfiniBench）：全精度平均 43.05、FA3-fp8 41.53（Retr.KV 7.0→0.4）、SageAttention2-8b 43.06（p.21）。
  - Table 18（L20）：per-thread 量化、兩層累加、smooth Q 的額外開銷 284→273 TOPS（p.23）。
- **跟 Tiara 的關係**〔判讀〕：
  - SageAttention 在 kernel 裡**即時**量化 Q、K（每 block 或每 thread）。所以如果縮圖已經存成 INT8（每 token 一個 scale），K 這邊可以直接餵 INT8 QKᵀ。但 Q 也要即時量化，品質要另驗。
  - 只用在「新問題在縮圖上 prefill 挑 block」這一步（LF §4.2），不是用在還原。
- **AMD**：只有 Triton 的 v1 理論上能在 ROCm 跑，作者說沒測、沒最佳化（GitHub issue #20 作者回覆；#125 使用者回報 1.0 分支可跑）〔二手〕。

#### H8：vLLM，*The State of FP8 KV-Cache and Attention Quantization in vLLM*（vLLM blog，2026-04-22）
- **讀了什麼**：〔文件，原始 HTML 文字檔核對〕<https://vllm.ai/blog/2026-04-22-fp8-kvcache>，查證 2026-10-27。
- **事實**：
  - `--kv-cache-dtype fp8` 會量化 KV，而且注意力的兩個矩陣乘（QK 與 ScoreV）都用 FP8（e4m3）。
  - Hopper 實驗是單張 H100、vLLM 自己 fork 的 FA3。
  - 128K 撈針：BF16 91%，FP8 13%。原因是 tensor core 裡「不精確的 FP32 累加」（softmax·V 的內積維就是 context 長度）。
  - 加上兩層累加（參考 SageAttention2）回到 89%，代價是暫存器壓力增加、變慢。
  - 單請求（H100、Llama-3.1-8B）：FP8 的 decode ITL 斜率是 BF16 的 54%，打平點約 7,010 token（v0.19.1；v0.10.2 是 63%、24,889）。
  - B200 上用 FlashInfer，原文說沒有累加問題。
  - 原文沒有 MI300X 的數字。
- **跟 Tiara 的關係**〔判讀〕：縮圖若存 FP8，可以不經 BF16 直接進 FA3。但 128K–256K 是我們的工作區間，正好是累加誤差出事的長度，一定要開兩層累加並自己驗。repo 的 E1 是「FP8 存、BF16 算」，不等於「FP8 算」（LF §4.2 已提）。

#### H9：MI300X 的 8 位元矩陣單元（CDNA3 白皮書；repo F3）
- **讀了什麼**：〔全文〕CDNA3 白皮書 p.7 表、p.24 表（pdftotext），<https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf>，查證 2026-10-27。
- **事實**：
  - 矩陣型別：FP64、FP32、TF32、FP16、BF16、FP8、INT8。全文沒有「INT4」〔全文〕。
  - MI300X：TF32 653.7、BF16 1,307.4、FP8 2,614.9、INT8 2,614.9 TOPS（稠密），FP32 向量 163.4 TF（p.24）。
  - p.7 的 TF32 列寫 490.3，和 p.24 的 MI300X 欄 653.7 不同。490.3 看起來是 MI300A 的數字（p.24 第一欄）〔判讀〕。
- **現有 kernel**：
  - vLLM 0.28 的 ROCM_ATTN 在 kernel 內把 FP8 cache 乘 k_scale 轉回，再用 BF16 算（`F3_dequant_fp8.md` §4.4 引 `chunked_prefill_paged_decode.py:216-217`）〔程式碼，repo 已記〕。
  - vLLM 的 ROCm 注意力後端部落格（2026-02-27）列 7 個後端，沒有寫任何 KV dtype 的支援或 FP8 注意力的數字；只有「shuffled KV layout 讓 decode 吞吐好 15–20%」，沒附設定〔文件〕。
  - AITER／CK 的 FP8 或 INT8 注意力實測 TFLOPS：**本次搜尋沒找到**（查詢：`AITER paged attention FP8 KV cache MI300X decode speedup`、`MI300X FP8 flash attention forward TFLOPS composable kernel aiter fmha fp8 benchmark`）。

### 2.3 低秩重建

#### H10：Sun et al., *ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference*（ICML'25；arXiv 2410.21465）
- **讀了什麼**：〔部分〕§4、附錄 A.6、B.1（p.5–6、p.18、p.20），<https://arxiv.org/pdf/2410.21465>，查證 2026-10-27。
- **一句話**：K 存成 pre-RoPE 的低秩，V 卸載到 CPU；decode 時只拿選中的 chunk，K 在 GPU 上重建。
- **kernel 的關鍵一步**：
  - 演算法 p.6：`K_sparse ← MatMul(Gather(A, I), B)` 之後套 RoPE。
  - 實作（B.1，p.20）：把「低秩重建 K」、「抓 V」等融合成 CUDA kernel，用多個 stream 讓重建 K 和抓 V 重疊。rank 160、chunk 8、稀疏預算 1.56%。
- **量到的數字**（Table 13，p.18；Llama-3-8B-1M、A100，一個 transformer block 的 decode）：
  - 重建 K（和抓 V 重疊）1.25–1.51 ms。
  - 抓 V 1.66–1.84 ms（48×64K 到 6×512K）。
  - → 重建完全藏在 PCIe 搬運後面。
  - 原文的等效頻寬算式（p.6）：A100 PCIe 31.5 GB/s、HBM 2 TB/s → 等效 7.2 TB/s。
- **跟 Tiara 的關係**〔判讀〕：
  - 在 GPU 閒時，「重建和下一塊搬運重疊」可以把重建藏起來。
  - repo X3 量到的翻轉，正是 GPU 忙時重建拉長到超過搬運（§3.3）。

#### H11：Chang et al., *Palu: KV-Cache Compression with Low-Rank Projection*（ICLR'25；arXiv 2407.21118）
- **讀了什麼**：〔部分〕§3.1、§4 效率（p.4、p.9–10），<https://arxiv.org/pdf/2407.21118>，查證 2026-10-27。
- **事實**：
  - K 在注意力 kernel 裡從低秩潛在向量即時重建；V 的基底併進 W_o。
  - RTX 4090、Llama-2-7B、50% 壓縮：RoPE 注意力模組在 64K 時 1.89×，V 再加 4-bit 量化 2.91×（p.9）。
  - 短序列幾乎沒有加速。
- **跟 Tiara 的關係**：屬於「換 attention」路線；基底吸收屬 agent G。

#### H12：Chang et al., *xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction*（arXiv 2503.18893）
- **讀了什麼**：〔部分〕§2.4、§5（p.4、p.6–7），<https://arxiv.org/pdf/2503.18893>，查證 2026-10-27。
- **事實**：
  - 單張 A100、Llama-3.1-8B、8 倍壓縮。
  - 選擇性重建：只重建選中的 token，成本不隨序列長度成長。
  - 照 ShadowKV 把重建和 V 的 PCIe 搬運重疊。
  - 數字只在 Fig. 5。
- **跟 Tiara 的關係**：repo 的 lr_recon（`xl:k20:v30:cal+q:int4_kivi`）就是這個格式（`B0_interference.md` L43）。

#### H13：repo 的重建 kernel 是怎麼做的（程式碼）
- `code/m16_xtime.py` L29、L39、L1621–1644、L1737：解碼＝係數反量化（`m16_xtime_kern.coef_dequant`，Triton）→ **BF16 `torch.matmul`**（ROCm 走 rocBLAS／hipBLASLt，見 `m16_xtime_kern.py` 檔頭）→ 融合收尾 kernel（加平均、K 再套 RoPE、寫進 KV 視圖，FP32 算再捨入成 BF16）。
- 形狀：每 4 層一組（`XKV_W = 4`，L119），係數 [T, r]，基底 [r, 4·H·128]。
  - Llama：N＝4096，k22 的聯合秩 r＝22×4×8＝704。
  - Qwen-7B：N＝2048，k11 的 r＝176（`m16_xtime_kern.py` 檔頭）。
  - X3 的 `xkv_g4_f<f>`：r＝4·H·128÷f（`m16_xtime.py` L30）。
- **數值**：GEMM 用 BF16 輸入、BF16 輸出（PyTorch 預設 FP32 累加）；收尾在 FP32。repo 沒有用 TF32 或 FP8 做重建（grep `tf32` 無結果）〔程式碼〕。
  - 〔判讀〕BF16 輸入的重建和原本 BF16 KV 是同一個精度等級，數值上可接受。
  - TF32 在 H200 和 MI300X 都只有 BF16 的一半速度（H200 989 含稀疏；MI300X 653.7），沒有理由用。
  - FP8 重建：基底、係數都要另外量化，品質未知（沒找到量過的論文）。

### 2.4 隔離（只補 repo 沒有的事實）

#### H14：CUDA Green Contexts（CUDA Driver API 13.4 參考）
- **讀了什麼**：〔文件，WebFetch 轉述〕<https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html>，查證 2026-10-27。
- **事實**：
  - 能切的資源：SM、workqueue 設定、既有 workqueue；**沒有提到記憶體頻寬**。
  - CC 9.0 以上 SM 數的對齊是 8（預設 coscheduled SM 數也是 8）。
  - 不保證不同 green context 的 kernel 會同時執行，也不保證能前進。
  - 兩種情況實際會用到比分配還多的 SM：MPS 的 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`；9.x 上用 dynamic parallelism 的 module。
  - 頁面沒寫哪一版 CUDA 開始支援〔未查證；常見說法是 12.4，本次沒核到〕。
- **與既有整理**：ShadowServe 認為 Green Context 表現差、不能切頻寬（`L8b_contention_aware_restore.md` L294–296；`L5c_compress_on_offload.md` L257）。Tutti 用它隔離 I/O kernel（L8b L82）。

#### H15：HIP CU mask 與 stream priority（HIP Runtime API 參考，docs-6.1.5）
- **讀了什麼**：〔文件，WebFetch 轉述〕<https://rocm.docs.amd.com/projects/HIP/en/docs-6.1.5/doxygen/html/group___stream.html>，查證 2026-10-27。
- **事實**：
  - `hipExtStreamCreateWithCUMask(stream, cuMaskSize, cuMask)`：每個 bit 代表一個 CU，前 32 bit 是前 32 個 CU；超過實體 CU 數的部分忽略；「使用者要自己確保輸入有意義」。
  - 文件沒說這個 mask 是硬性限制還是提示〔文件〕。
  - stream priority：數字越小優先權越高；超出範圍會被夾住。
- **repo 實測**（11 §3d）：留 3/4 CU 給還原只對 Cake 有用、背景慢 12–20%；io_prio 沒效果。

---

## §3 反量化／重建在哪些情況才會變成瓶頸〔算術〕

**一句話**：反量化是「讀小寫大」的頻寬型工作，在 HBM 上一塊只要幾十微秒，PCIe 永遠比它慢 4 倍以上；低秩重建在秩大時是算力型、秩小時是頻寬型，但**兩者在 GPU 忙時都不是被 FLOPs 拖慢，而是被「排不到 CU／SM」拖慢**。

### 3.1 記號與代入值
- 一塊＝512 token。Llama-3.1-8B 一塊 BF16＝32 層×2×8 head×512×128×2 B＝64 MiB；Qwen-7B 28 MiB（`m10_dqsweep.py` `cmd_kernels`，`tmax`、`lo:hi`）。
- INT4 比 BF16 小 3.2 倍（`code/s3_mech_arith.py` L50 `INT4_RATIO`；X3 micro 的 int4_kivi 位元組 671,088,640／2,147,483,648＝1/3.2）。
- H200：pinned H2D 53.55 GiB/s、BF16 matmul 788.1 TFLOPS、HBM 裝置內複製 4,173.9 GB/s（`results/RUNLOG_H200.md:63`）；matmul÷HBM＝189 FLOP／B（`arith.md` L104）。FP32 CUDA core 67 TFLOPS、HBM 4.8 TB/s〔文件〕。
- MI300X：BF16 1,307.4 TF、FP32 向量 163.4 TF、HBM 5.3 TB/s〔文件〕→ 矩陣平衡點 247、向量平衡點 31 FLOP／B〔算術〕。

### 3.2 反量化
- **位元組**：每個 BF16 輸出讀 0.5（INT4）或 1（INT8／FP8）byte＋scale，寫 2 byte。
- **運算**：每元素 2–5 個 ALU 運算（QServe p.10）→ 強度約 0.8–2 ops／B ≪ 14（H200 CUDA core）／31（MI300X 向量）→ **頻寬型**。
- **一塊的時間**（Llama 64 MiB）：
  - 下限＝(20＋64) MiB ÷ 4.8 TB/s＝0.018 ms（INT4，H200）。
  - 實測：H200 k2（torch.compile）int4_token 閒 0.066、忙 0.028 ms；融合 k1 int4_kivi 閒 0.073、忙 0.049 ms（`kernels_llama.csv`）。
  - 換算有效頻寬：閒 1.3 TB/s（27%）、忙 3.0 TB/s（63%）。(16.8＋0.5＋67.1) MB÷0.0279 ms≈3.0 TB/s。
  - **H200 上閒時比忙時慢**：一致出現在所有格式（例如 fp8_token k2 閒 0.0596 vs 忙 0.0298 ms）。原因 repo 沒查〔判讀：可能是閒時 GPU 降頻或電源狀態，未驗證〕。用這組數字做判定前要先弄清楚。
- **對比搬運**：同一塊 INT4 過 PCIe＝20 MiB÷(53.55×1024 MiB/s)＝0.365 ms；BF16＝1.167 ms。
  - 反量化／搬運＝0.028–0.073÷0.365＝**8–20%**（H200）。
  - MI300X X3 micro：32 塊 int4_kivi 一次拆 1.69 ms 對 H2D 13.68 ms＝12%（`results/m16_stage3_mi300x/x3_time/micro_llama.csv`）。
- **重疊**：第 i 塊反量化和第 i＋1 塊 H2D 同時跑時，總時間≈n·t_H2D＋t_dq（最後一塊）。只要 t_dq < t_H2D，反量化就只剩最後一塊的 0.03–0.07 ms。
  - **但** repo 11 §3d 在 MI300X 上發現：GPU 忙時「每塊開一次 kernel」要和背景排隊，反而是「全部搬完一次拆」（S4）最快。重疊和「少開 kernel」要權衡，而且這個權衡隨 GPU 忙碌改變〔判讀〕。

### 3.3 低秩重建（係數 C[T×r] × 基底 B[r×N]）
- **強度**：每個 BF16 輸出元素做 2r FLOP，寫 2 byte → **r FLOP／B**（讀係數是每 token r 個值、被 N 個輸出共用，N≫r 時可忽略）〔算術〕。
  - 和平衡點比（H200 189、MI300X 247）：

    | 設定 | r | 判斷 |
    |:--|--:|:--|
    | 每 head 低秩 lr_r16–r64 | 16–64 | 頻寬型 |
    | Qwen xKV k11 | 176 | 邊界 |
    | xKV f8（Llama）／k22 | 512／704 | 算力型 |
    | 任務書提的 R≈544–832 | 544–832 | 算力型，是平衡點的 2.9–4.4 倍（H200） |

- **MI300X 實測效率**〔算術，由 X3 micro 換算；含收尾 kernel，所以是 GEMM 效率的下界〕：
  - 32 塊＝16,384 token。
  - Llama xkv_g4_f8（r＝512，N＝4096，8 組×K／V）：FLOPs＝2·16384·512·4096·16＝1.10e12；閒 3.96 ms → 278 TFLOPS（BF16 峰值的 21%）。
  - Llama xkv_g4_f4（r＝1024）：2.20e12 ÷ 5.91 ms＝372 TFLOPS（28%）。
  - Qwen xkv_g4_f8（r＝256，N＝2048，7 組）：2.40e11 ÷ 1.38 ms＝174 TFLOPS（13%）。
  - 收尾 kernel 至少要讀寫約 4–6 GiB（GEMM 輸出寫一次，收尾讀一次再寫一次），在 5.3 TB/s 下約 0.8–1.2 ms〔算術〕。扣掉後 f8 的 GEMM 約 2.8–3.2 ms → 約 340–390 TFLOPS（26–30%）〔算術，推估〕。
- **形狀造成的效率損失**〔判讀〕：K 維＝r（500–1000）很短，每個輸出 tile 只跑 r／64≈8–16 次 K 迴圈，prologue／epilogue（載入、寫回）占比大。cuBLAS／hipBLASLt 對這種「M 大、N 中、K 小」的形狀沒有公開的效率數字，**本次沒找到**（查詢：skinny GEMM small K tensor core efficiency hipBLASLt）。
- **GPU 忙時**（MI300X X3 micro，duty 0→0.9，同卡算力型背景）：

  | 格式 | 閒 (ms) | 忙 (ms) | 倍數 | 秩 |
  |:--|--:|--:|--:|--:|
  | int4_kivi 反量化 | 1.69 | 2.43 | 1.44× | — |
  | xkv_g4_f8 | 3.96 | 7.08 | 1.79× | 512 |
  | xkv_g4_f4 | 5.91 | 18.33 | 3.10× | 1024 |
  | lr_r16（每 head） | 2.34 | 12.41 | 5.30× | 16 |
  | lr_r64（每 head） | 2.48 | 11.96 | 4.82× | 64 |

  Qwen 同型：lr_r16 1.02→5.26（5.2×）、xkv f8 1.38→3.35（2.4×）、int4_kivi 0.59→1.55（2.6×）。
  - 〔判讀〕**FLOPs 最少的 lr_r16 反而慢最多**。所以 GPU 忙時拖慢重建的不是 tensor core 被搶，而是 kernel 的 workgroup 要和背景搶 CU。lr 是 L×H＝256 個小批次 GEMM（K＝16），workgroup 多而短。這和 repo 11 §3d 的推論「瓶頸是每次開 kernel 都要排隊」一致。
  - `experiment_log.md` L157 的 E-L3（Llama 2.79–2.80×）是同一批資料的摘要值，本表列的是各格式的原始中位數。
- **H200**（repo B0／G-M08，背景是真正的長 context decode，頻寬型）：
  - lr_recon 的速度只剩閒時的 0.62–0.73（Llama，decode 4×129K）、0.39–0.53（Qwen，decode 4–6×258K）。
  - 背景 decode 吞吐同時只剩單獨跑時的 0.41–0.74（慢 26–59%；`b0_report.json` `m08` 的 lr_recon 格）〔複核 2026-10-27：原寫「吞吐掉到 0.26–0.59」，但那是 `bg_slowdown`＝1−吞吐比（`code/s3/b0.py:807` `slow = 1 - v["bg_ratio"]`），改為吞吐比 0.41–0.74〕。
  - stream 和 stream_hi 幾乎一樣（Llama 0.628 vs 0.637）。
  - 〔判讀〕這和 MI300X「頻寬型背景（大陣列反覆複製）下幾乎不受影響」（results_analysis.md L14）不一致：真的 decode 注意力和 copy 迴圈不是同一種背景。repo 的「只有算力型背景才翻轉」要限定在「copy 迴圈當頻寬型背景」這個設定。

### 3.4 什麼時候反量化／重建會變成還原的瓶頸（總結）〔算術＋判讀〕
- **反量化**：t_dq ≥ t_H2D(壓縮後) 才會。H200 上要慢約 5–13 倍（0.365÷0.028–0.073），目前量過的爭用最多 2.6 倍 → **不會**。例外：每塊開一次 kernel、GPU 又很忙時，排隊時間（不是運算）可能讓總時間變長（MI300X S0 vs S4）。
- **低秩重建**（MI300X X3 micro，Llama 32 塊，搬運＋解碼兩段相加、不重疊；`micro_llama.csv`）：

  | 格式 | 閒：H2D＋解碼 (ms) | duty 0.9：H2D＋解碼 (ms) |
  |:--|--:|--:|
  | BF16 原樣 | 39.67＋0 | 64.85＋0 |
  | INT4 KIVI（一次堆疊 copy＋一次拆） | 13.68＋1.69＝15.37 | 14.92＋2.43＝17.35 |
  | xkv_g4_f8（r＝512） | 7.30＋3.96＝**11.26** | 23.71＋7.08＝30.79 |
  | xkv_g4_f4（r＝1024） | 13.31＋5.91＝19.22 | 29.06＋18.33＝47.39 |

  - 閒時 xKV 最快，忙時 INT4 最快 → 這就是 repo 的「翻轉」。
  - 〔判讀〕忙時被拖慢的不只是重建：xKV 的 H2D（`h2d_region`，多段分開 copy）也從 7.30 變成 23.71 ms（3.2×）。INT4 的 H2D 是一次堆疊 copy（`h2d_stack`），只慢 1.09×。
  - 兩邊共同的規律是「操作次數多的那條路，忙時慢得多」，和 §3.3 的 lr_r16 一致。
  - 重建本身在 duty 0.9 時仍短於自己的 H2D（f8 0.30、f4 0.63；E-L3 記 0.30–0.37）。所以在這組量測裡，翻轉不是「重建比搬運久」造成的，而是兩段都被排隊放大後加總。
  - 若改成重疊執行（ShadowKV 式），時間會變成 max(H2D, 解碼) 而不是相加。這個版本 repo 沒量。

---

## §4 判讀

**一句話**：硬體面能做的事，排序是：把重建做成少數幾個大 kernel → 給還原自己的 SM／CU 份額（MPS 已證明有效）→ 在縮圖挑塊時用 FP8／INT8 tensor core。位元技巧和 CPU 端解壓都不值得做。

### 4.1 H200 上最快的 KV 反量化路徑
- **已量到的最快**：torch.compile（k2）或 repo 融合 Triton（k1／k3:8:2）。Llama 一塊 64 MiB 0.028–0.087 ms，最多到 HBM 峰值的約 70%（fp8_token k2 忙時：97 MiB÷0.0298 ms≈3.4 TB/s）〔算術〕。
- 再往上只剩約 1.4 倍的空間，對總還原時間影響 <5%（反量化只佔搬運的 8–20%，而且可以重疊）〔算術〕。
- **更快的方向不是反量化本身**，而是不寫回 BF16：BitDecoding（INT4 → 暫存器 FP16 → `wgmma`）或 FA3-FP8。但都要換 attention 路徑；FP8 在 128K 有累加陷阱（H8）。

### 4.2 MI300X 上最快的 KV 反量化路徑
- repo F3：torch.compile／Triton，INT8 0.031、FP8 0.026、INT4 0.027 ms／64 MiB（cold），約 3.3 TB/s、峰值 62%（`F3_dequant_fp8.md` §4.1）。
- 低精度直接算注意力：硬體有 FP8（FNUZ）、INT8 MFMA；但**沒找到量過的注意力 kernel**。vLLM 0.28 的 ROCM_ATTN 在 kernel 內轉回 BF16 算，不是 FP8 運算（F3 §4.4）。
- P07 §4 記的 Petit（AMD 部落格）：FP4→BF16 用位元組轉換指令，從 31 條降到 15 條，是 MI300X 上唯一找到的位元技巧資料（C 級，本次沒重讀）。

### 4.3 低位元 QKᵀ（不先反量化）可不可用

| | H200 | MI300X |
|:--|:--|:--|
| FP8 | **可**：`wgmma` e4m3／e5m2；FA3、vLLM FP8 KV；長 context 要兩層累加 | 硬體可（FNUZ）；現成注意力 kernel 的實測沒找到 |
| INT8 | **可**：`wgmma` s8；SageAttention2-8b 速度＝FA3-fp8 | 硬體可；沒找到現成 kernel（SageAttention 只有未測的 Triton v1） |
| INT4 | **不可**：沒有 s4 `wgmma`，要先轉 FP16（BitDecoding） | **不可**：矩陣型別沒有 INT4 |

### 4.4 怎麼把重建 GEMM 和爭用隔開
- **最便宜、證據最強**：少開 kernel。MI300X 上「一次拆」把忙時的拆從 9.7–23.4 降到 1.4–4.3 ms（11 §3d）。
  - 對重建：把每 head 的低秩（256 個小 GEMM）換成跨層的大 GEMM（xKV），忙時倍數從 5.3× 降到 1.8–3.1×（§3.3 表）〔判讀：同時改了秩，不是乾淨的對照〕。
  - 進一步：重建＋RoPE＋寫回做成一個 persistent kernel，只排一次隊〔判讀，未實作〕。
- **SM 份額**：
  - H200 MPS 已量過有效（GB0-SHARE，mps87 4.28×）。
  - Green Context 只切 SM、不切頻寬、不保證同時執行（H14），H200 上沒量過。
  - stream priority 沒效（B0：stream vs stream_hi 相同）。
  - MI300X CU mask：API 沒說是不是硬性；repo 量到只對 Cake 有效。
- **搬到別處**：
  - copy engine 不能做轉換（Hopper、CDNA3 的白皮書／文件都沒有這個功能，LF F13、§1）。
  - Blackwell 的解壓引擎只吃無損格式。
  - CPU 端見 4.5。

### 4.5 CPU 端反量化（CPU／SSD 層）〔算術〕
**一句話**：在 PCIe 前就把 INT4 解成 BF16，讓 PCIe 位元組變 3.2 倍、DRAM 流量變 7–10 倍；在我們的平台上不會贏。

- **GPU 路徑**（每塊 Llama 64 MiB）：
  - DRAM 讀 20 MiB → PCIe 20 MiB（0.365 ms）→ GPU 反量化 0.03–0.07 ms。
  - 不重疊合計約 0.39–0.44 ms。
- **CPU 路徑**：
  - CPU 讀 20 MiB、寫 64 MiB BF16 到 pinned buffer（寫入若觸發 read-for-ownership 再加 64 MiB）→ DMA 讀 64 MiB → PCIe 64 MiB（1.167 ms）。
  - DRAM 流量 148–212 MiB。每個 job 拿到的 DRAM 頻寬 98–125 GB/s（LF F4）→ 1.24–2.27 ms。
  - 合計 **≥1.24 ms**，是 GPU 路徑的約 3–5 倍。
- **CPU 路徑要贏的條件**：GPU 反量化 ≥ (1−1/3.2)×1.167＝0.80 ms／塊（不重疊時）；若和搬運重疊，則要 ≥1.167 ms。
  - 這是 H200 實測（0.028–0.087 ms）的 9–29 倍。
  - repo 量過最大的爭用倍數：MI300X 融合版 1.44–2.6×（X3），逐塊 PyTorch 版 2.4–3.8×（L2 L19）。
  - → **不會發生**。
- **CPU 的 SIMD 不是瓶頸**（粗估，未實測）：
  - Zen 4 上 `VPERMB`、`VPSHUFB`、`VPMOVZXBW`（zmm）吞吐都是 1 條／週期；`VCVTNE2PS2BF16`（兩個 FP32 zmm → 32 個 BF16）實測 1.35 週期／條（uops.info）〔全文〕。
  - 假設每 32 個輸出約 8 條 zmm 指令（解包 2–3、轉 FP32 2、乘 scale 2、轉 BF16 1），每條約 1 週期 → 每核約 8 B／週期 → 3.1 GHz 下約 25 GB/s／核，16 核約 400 GB/s ≫ DRAM 的 98–125 GB/s〔算術，指令數是假設〕。
  - 所以**卡住的是 DRAM 和 PCIe，不是 AVX-512**。換成 AMX 也沒用（AMX 是矩陣乘，不是解包）。
- **唯一可能的例外**〔判讀〕：GPU 完全不能給還原任何 SM（例如被 MPS 限死在 0%）。這不是我們會設定的情況。另外，MI300X＋vLLM 0.28 的 4.1 GiB/s 慢路徑（L7c）是每塊 copy 的固定成本，位元組變 3.2 倍只會更慢。
- **CPU 端解碼真正有用的地方**：無損層（zstd、LZ4）解出來的是**量化後的整數**，不是 BF16。這時 CPU 只做熵解碼，PCIe 上搬的仍是 INT4。repo S1 的 CPU 解壓 0.5–2.1 GB/s 是這一類（LF §4.1）。

### 4.6 對第三階段的建議（只列，不評估）
1. B1「還原給多少 SM」：用 MPS（已證明）當主旋鈕；Green Context 當同 process 的替代，先量它在 H200 上的隔離度。
2. 縮圖的重建：優先跨層大 GEMM＋融合收尾；忙時倍數和 kernel 數一起記錄，才分得出「算力」和「排隊」。
3. 縮圖上挑塊的那一步：試 FP8（開兩層累加）或 INT8（SageAttention2-8b）；這是唯一能用到 tensor core 低精度的地方。
4. 不做：CPU 端反量化、AVX-512／AMX 解包、新的 GPU 位元技巧。

---

## §5 查不到／未查證

| 項目 | 查詢字串 | 狀態 |
|:--|:--|:--|
| Hopper 上 `mma.sync .s4` 是不是模擬、速度多少 | `H100 INT4 tensor core not supported mma s4 sm_90 emulated`；PTX ISA 全文搜尋 `emulat` | PTX ISA 沒寫；只確定 `wgmma` 沒有 s4，SageAttention2 說 Hopper 沒有原生 INT4 |
| MI300X FP8／INT8 注意力實測 TFLOPS（AITER、CK） | 見 H9 | 本次搜尋沒找到 |
| SageAttention 的 MI300X port | `SageAttention ROCm MI300X port`；`gh search issues --repo thu-ml/SageAttention ROCm` | 只有未測的 Triton v1 |
| Zen 4／SPR 上 int4→bf16 解包的實測 GB/s | `AVX-512 int4 to bf16 dequantization throughput GB/s Zen 4 OR Sapphire Rapids VPERMB` | 本次搜尋沒找到；§4.5 只做指令吞吐粗估 |
| llama.cpp／xFasterTransformer／KTransformers／oneDNN 的 int4 解包 GB/s | 沒有逐一查 | 〔未查證〕（KTransformers 的 AMX 數字見 LF F8，是 GEMM 不是解包） |
| skinny GEMM（K＝r≈500–1000）的 tensor core 效率 | 見 §3.3 | 沒找到；只能由 repo X3 換算 21–28%（含收尾，MI300X） |
| Green Context 首次出現的 CUDA 版本、H200 上的隔離實測 | — | 文件頁沒寫版本；沒有 H200 還原的量測 |
| CU mask 是否為硬性限制 | HIP 文件 | 文件沒寫 |
| TensorRT-LLM INT8／FP8 KV 在注意力裡怎麼反量化 | `TensorRT-LLM KV cache quantization ... dequantize in attention kernel` | 只確認文件列有 INT8／FP8／NVFP4 KV〔文件，搜尋摘要〕 |
| FlashInfer FP8 KV decode 是否在 kernel 內轉 FP16 | 沒查 | 〔未查證〕 |
| 任務書提的「vLLM v0.32 `int4_per_token_head`」 | `vllm kv_cache_dtype int4_per_token_head` | 搜到的文件只列 `int8_per_token_head`、`fp8_per_token_head`（TRITON_ATTN，v0.21 文件）；repo F3 記 MI300X 上 vLLM 0.28 的 `int4_per_token_head` 走 TRITON_ATTN（F3 §4.3）。v0.32 本次沒核 |
| H200 閒時反量化比忙時慢 | — | repo 資料如此（§3.2），原因未查 |
| LUT-GEMM 的 GPU 型號 | — | 只讀到摘要 |

---

## 參考

### 論文
- [P1] Y. J. Kim *et al.*, "Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production," arXiv 2211.10017, 2022. <https://arxiv.org/pdf/2211.10017>
- [P2] E. Frantar *et al.*, "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models," arXiv 2408.11743. <https://arxiv.org/pdf/2408.11743>
- [P3] Y. Lin *et al.*, "QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving," MLSys 2025, arXiv 2405.04532. <https://arxiv.org/pdf/2405.04532>
- [P4] D. Du *et al.*, "BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache," arXiv 2503.18773. <https://arxiv.org/pdf/2503.18773>
- [P5] J. Zhang *et al.*, "SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization," arXiv 2411.10958. <https://arxiv.org/pdf/2411.10958>
- [P6] J. Zhang *et al.*, "SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-bit Training," arXiv 2505.11594（只用到 RTX 5090 1038 TOPS 一句，p.1）. <https://arxiv.org/pdf/2505.11594>
- [P7] H. Guo *et al.*, "Fast Matrix Multiplications for Lookup Table-Quantized LLMs (FLUTE)," arXiv 2407.10960. <https://arxiv.org/pdf/2407.10960>
- [P8] G. Park *et al.*, "LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models," ICLR 2024, arXiv 2206.09557. <https://arxiv.org/pdf/2206.09557>
- [P9] L. Wang *et al.*, "Ladder: Enabling Efficient Low-Precision Deep Learning Computing through Hardware-aware Tensor Transformation," OSDI 2024. <https://www.usenix.org/system/files/osdi24-wang-lei.pdf>
- [P10] Z. Liu *et al.*, "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache," arXiv 2402.02750. <https://arxiv.org/pdf/2402.02750>
- [P11] H. Sun *et al.*, "ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference," arXiv 2410.21465. <https://arxiv.org/pdf/2410.21465>
- [P12] C.-C. Chang *et al.*, "Palu: KV-Cache Compression with Low-Rank Projection," arXiv 2407.21118. <https://arxiv.org/pdf/2407.21118>
- [P13] C.-C. Chang *et al.*, "xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction," arXiv 2503.18893. <https://arxiv.org/pdf/2503.18893>
- [P14] J. Wei *et al.*, "T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge," arXiv 2407.00088（只確認它用 AVX2 `PSHUF`／NEON `TBL`，平台是邊緣裝置；本檔未採用數字）. <https://arxiv.org/pdf/2407.00088>

### 廠商文件與程式碼
- [V1] NVIDIA, *PTX ISA 9.4*（`wgmma.mma_async` 型別表與語法）. <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- [V2] NVIDIA, H200 product page & spec table. <https://www.nvidia.com/en-us/data-center/h200/>
- [V3] NVIDIA, CUDA Driver API, *Green Contexts*. <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html>
- [V4] NVIDIA TensorRT-LLM, `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h`（main，查證 2026-10-27）. <https://github.com/NVIDIA/TensorRT-LLM>
- [V5] L. Wilkinson, "Introducing Machete, a mixed-input GEMM kernel optimized for NVIDIA Hopper GPUs," Red Hat Developer, 2024-10-14. <https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel>
- [V6] vLLM, "The State of FP8 KV-Cache and Attention Quantization in vLLM," 2026-04-22. <https://vllm.ai/blog/2026-04-22-fp8-kvcache>
- [V7] vLLM, "Beyond Porting: How vLLM Orchestrates High-Performance Inference on AMD ROCm," 2026-02-27. <https://vllm.ai/blog/2026-02-27-rocm-attention-backend>
- [V8] AMD, *Introducing AMD CDNA 3 Architecture* white paper, pp. 7, 24. <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf>
- [V9] AMD, HIP Runtime API, Stream Management（docs-6.1.5）. <https://rocm.docs.amd.com/projects/HIP/en/docs-6.1.5/doxygen/html/group___stream.html>
- [V10] uops.info：VPERMB、VPSHUFB、VPMOVZXBW、VCVTNE2PS2BF16、VPANDQ（ZMM）. <https://uops.info/html-instr/VPERMB_ZMM_ZMM_ZMM.html>（其餘同路徑）
- [V11] thu-ml/SageAttention GitHub issues #20、#125. <https://github.com/thu-ml/SageAttention/issues/20>

### 本專案內部
- `docs/research_20261010_followup/F3_dequant_fp8.md` §0、§3、§4.1、§4.4（MI300X 反量化 0.031 ms，`code/m9_f3_dequant.py`，run `20261010-074630-f3-dequant`）
- `code/m10_dequant_fast.py`（k1＝`decode_fast` rows 16；k2＝torch.compile；k3:<rows>:<warps>；k4＝FP8 位元運算不查表；`decode_variant` L365–393）；`code/m10_dqsweep.py` `cmd_kernels`
- `results/m10_sweep_h200/dqsweep/A/kernels_llama.csv`、`kernels_qwen.csv`、`choice_*.json`（H200 run `20261018-145622/145731-m10h-dqs-A-*`）
- `results/RUNLOG_H200_S3.md` S3-C0-v2（L477–484）；`docs/stage3_design_20261024/runs/C0_cost_model.md:63`
- `code/m16_xtime.py`、`code/m16_xtime_kern.py`；`results/m16_stage3_mi300x/x3_time/micro_llama.csv`、`micro_qwen.csv`（runs `20261020-212244/213232-m16-x3-micro-*`）
- `results/s3_h200/b0/b0_report.json`；`docs/stage3_design_20261024/runs/B0_interference.md`
- `docs/precision_20261013/11_issues.md` §3d、§4；`docs/precision_20261013/cards/P07_sweep_formats_sys_dequant.md` §4
- `docs/stage3_20261019/results_analysis.md` L14、L325；`docs/stage3_20261019/experiment_log.md` L157
- `docs/stage3_design_20261024/lit/L8b_contention_aware_restore.md`；`docs/stage3_20261019/lit/L5c_compress_on_offload.md` L255–265；`L2_compression_offload_prefetch.md` L19
- `docs/linalg_accel_20261027/lit/LF_hw_isa.md` F4、F9、F10、F13、F16、§4
- `results/RUNLOG_H200.md:63`；`docs/stage3_design_20261024/mechanisms/arith.md` L104；`code/s3_mech_arith.py` L50
