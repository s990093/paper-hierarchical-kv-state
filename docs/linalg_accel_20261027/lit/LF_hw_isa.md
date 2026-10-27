# LF：硬體指令集與固定功能單元，會不會改變 KV 還原的成本（CPU：AMX、Zen 4／5 AVX-512；GPU：Hopper、Blackwell、CDNA3、CDNA4）

> 文獻 agent F，2026-10-27。規則見 `RULES.md`。只做查證與算術，沒有在 Mac 上跑任何實驗。
> 證據等級：〔全文〕〔部分〕〔摘要〕〔程式碼〕〔文件〕〔二手〕〔算術〕〔判讀〕〔未查證〕。**廠商峰值一律標〔文件〕，不是量測**；量測只抄原始來源，並附設定。
> 已有整理、這裡不重抄：Intel KV Shrink（QAT／IAA／DSA）與 MI300X 的 SDMA／hipCOMP／hipFile → `docs/stage3_20261019/lit/L0_intel_suite_and_amd.md`；MI300X 還原路徑（vLLM 4.1 GiB/s 的原因、zero-copy kernel、ROCm 批次 copy PR）→ `docs/stage3_20261019/lit/L7c_recheck_amd_restore_path.md`。

---

## §0 一頁結論（白話）

**一句話**：查完四家硬體，**沒有哪個 CPU 指令能讓「在 CPU 上直接算注意力」（R2）在我們的平台翻盤**；真正可能改變還原成本的是三件 GPU 端的事：**(1) GPU 端無損解壓（不是 CPU 解壓）、(2) 回來那一輪「用新問題在縮圖上算分數」這一步是 tensor core 算力型的，FP8 可以讓它便宜約 1.6 倍、(3) 2:4 稀疏能把縮圖的位元組再砍到約 56%，但條件很硬，而且壓 K 很傷品質。**

白話分四點：

1. **R2 的 3.2 TFLOPS 假設其實是對的。** EPYC 9554（Zen 4）每個核心每個時脈做 64 個 BF16 運算（`VDPBF16PS` 512-bit 指令，實測每時脈 1 條，uops.info）〔全文〕。16 核×64×3.1 GHz＝3.17 TFLOPS，正好是 repo 用的數字；用全核加速 3.75 GHz 算是 3.84 TFLOPS，打平的問題長度從 7／11 個 token 變成 9／13 個。**結論不變。**
2. **記憶體頻寬比算力更早卡住，但這反而對 CPU 有利一點點**：注意力對每個位元組的 KV 只做「GQA 組大小 × 問題長度」次運算（Qwen-7B 是 7q，14B 是 5q）。所以問題只有 1–4 個 token 時，CPU 是被 DRAM 頻寬卡住，不是算力。16 個核心大約拿得到 100–125 GB/s〔算術＋二手〕，比 PCIe（實測 57.5 GB/s）快，所以**問題很短時 CPU 注意力確實比搬 KV 快（0.12–0.15 s 對 0.26 s）**；但問題一超過 7–13 個 token，換成算力卡住，就輸了。我們的問題長度是 128–512 個 token（arith §6、§9），差很遠。
3. **換成有 AMX 的 Intel Xeon 也翻不了盤**：AMX 每核每時脈峰值 1024 個 BF16 運算（Intel 手冊 Table 20-2 推得）〔文件＋算術〕，是 Zen 4 的 16 倍；但實測只拿到峰值的 7%（PyTorch／oneDNN）到 29%（KTransformers 自己寫的 kernel）〔全文〕。用實測換算，16 核的打平點變成 23–33 個 token；就算用整顆 Granite Rapids（LIA 實測 40 TFLOPS）也只到 99–139 個 token，還是比 128–512 小或相當。**而且我們的 H200 節點根本沒有 AMX**（是 EPYC）。還有一點 R2 沒寫：回答的每一個生成 token 都要再讀一次全部 KV（每個 token 0.12–0.15 s），所以「不搬 KV」在 decode 階段更糟〔判讀〕。
4. **GPU 端可能改變決定的三件事**：
   - **GPU 端無損解壓**：repo 之前判定「無損壓縮在 CPU 層會大幅變慢」，前提是**CPU 解壓只有 0.5–2 GB/s**。如果解壓搬到 GPU（nvCOMP ANS／DietGPU 在 H100 上有人量到 480–592 GB/s〔部分，單一作者預印本〕），用 repo 量到的壓縮比 1.47 算，CPU 層的搬運時間最多可以省 32%（解壓與傳輸重疊）或 20%（不重疊）〔算術〕。**這是唯一可能翻掉既有判定的一點**，但沒人在 KV 上量過，H200 上要先量。
   - **FP8 tensor core**：「新問題在縮圖上 prefill 挑塊」那一步，問題 128–512 token 時每位元組做 500–3,600 次運算，遠高於 H200 的平衡點 189（arith §8），所以是**算力型**；FA3 在 H100 量到 FP8 1.2 PFLOPS 對 FP16 740 TFLOPS（約 1.6 倍）〔全文〕〔複核 2026-10-27：原寫「BF16 740 TFLOPS」，改為「FP16」；FA3 摘要（arXiv 2407.08608v2 p.1）寫的是 FP16，全文 pdftotext 無 BF16 字樣〕。品質要另外驗。
   - **2:4 稀疏 tensor core**：H200 與 MI300X 都有，條件是「每 4 個連續元素（沿著內積那一維）恰好 2 個是 0」，而且**只能是第一個運算元**。HieraSparse 把 K、V 當稀疏運算元，在 L40S／A100 上注意力快 1.3–1.8 倍；但 prefill 時壓 K 讓 LongBench 掉 11.1 分、壓 V 只掉 1.8 分〔全文〕。沒有 H100／H200／MI300X 的數字。

**跟還原無關、可以不管的**：Blackwell 的硬體解壓引擎（只有 B200／B300／GB200／GB300，且只吃 LZ4／Snappy／Deflate）、CDNA4 的 MXFP4／MXFP6（我們沒有 MI355X）、EPYC 9684X 的 3D V-Cache（每顆 CCD 96 MB，KV 是幾十 GB，讀一次就走）、Intel DSA／IAA（EPYC 沒有）、CXL 記憶體（實測只拿到理論頻寬的 20–47%，比 PCIe 還慢）。

**查不到的**：H200／H100 的 copy engine 數量（沒有官方文件；repo 只量到 H2D 與 D2H 同時跑時 H2D 53.40 GiB/s，表示兩方向至少各有一個引擎）、3090 伺服器的 CPU 型號（repo 沒記）、IAA 單顆的解壓 GB/s、Genoa 每顆 CCD 的官方頻寬、AMX 實際運作時脈。

---

## §1 總表

「我們有沒有」：H200 節點＝8×H200＋2×EPYC 9554（job 16 核）；MI300X 主機＝2×EPYC 9684X（容器 32 核）；3090 伺服器 CPU 型號〔未查證〕。

| 硬體 | 指令／單元 | 做什麼 | 官方峰值（出處） | 實測（出處與設定） | 條件 | 跟 Tiara 還原的關係 | 我們有沒有 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| Intel Xeon SPR／EMR／GNR | AMX（TMUL、8 個 tile；`TDPBF16PS`） | 矩陣乘（BF16、INT8；GNR 加 FP16） | 每個 `TDP*` 吞吐 16 cycles、延遲 52 cycles〔文件〕Intel Opt. Manual 248966-051 Table 20-2；一條 BF16 最大 16×16×32＝8,192 MAC → **1,024 BF16 FLOP／週期／核**〔算術〕 | KTransformers：單顆 Xeon 8452Y（36 核）MoE 層 21.3 TFLOPS（理論 73.7 的 29%）；PyTorch AMX 5.4 TFLOPS（7%）〔全文〕。LIA：SPR 20、GNR 40 TFLOPS 矩陣乘〔摘要〕 | 每個 tile 的 M≤16 列；問題太短（M＝q·G<16）會浪費；AVX-512 在「每專家 ≤4 token」時比 AMX 快（KTransformers §3.2） | R2 打平點 7–11 → 23–33（實測換算 16 核）／81–114（峰值）token；仍 < 128 | **沒有**（H200、MI300X 主機都是 EPYC） |
| Intel 未來（Diamond Rapids） | AMX-FP8、AMX-AVX512、AMX-MOVRS；AMX-COMPLEX（GNR-D、DMR） | FP8 矩陣乘等 | 只有 CPUID／指令列表，無吞吐數字〔文件〕ISA Ext. Ref. 319433-062 Table 1-2；AMX-TF32 與 AMX-TRANSPOSE 已從文件刪除（同文件修訂紀錄） | 無 | 未出貨 | 無 | 沒有 |
| AMD Zen 4（EPYC 9004：9554、9684X） | AVX-512（512-bit 拆兩半走 256-bit 管線）、`VDPBF16PS`、VNNI | 向量 BF16 內積、INT8 內積 | AMD 文件值：`VDPBF16PS zmm` 吞吐 1／週期（uops.info 轉載的 AMD 文件欄）→ **64 BF16 FLOP／週期／核**〔算術〕 | uops.info 實測 Zen 4：`VDPBF16PS zmm` 與 `VFMADD231PS zmm` 都是每週期 1 條（Zen 5 為 0.5 週期）〔全文〕 | 無特殊條件；BF16 乘、FP32 累加 | R2 的 3.2 TFLOPS 假設＝16×64×3.1 GHz，**來源成立** | **有**（兩台主機） |
| AMD Zen 5（EPYC 9005） | AVX-512「full 512b data path」 | 同上，寬度加倍 | AMD 新聞稿 2024-10-10〔文件〕；uops.info Zen 5 `VDPBF16PS zmm` 0.5 週期／條 → 128 FLOP／週期／核〔算術〕 | Phoronix：EPYC 9755 512-bit 模式比 256-bit 模式快 1.12×（應用平均）〔二手〕 | 部分 BIOS 可切 256／512 | 打平點約加倍（16 核 7.9 → 15.8 token） | 沒有 |
| EPYC 9554 記憶體 | 12 ch DDR5-4800 | 頻寬 | 460.8 GB/s／socket〔文件〕AMD 產品頁 | STREAM Triad **390 GB/s**（1×9554P、12×32 GB 2Rx8、NPS4、SMT off）〔全文〕Fujitsu RX1440 M2 報告 v1.9 p.20；每顆 CCD 讀 GMI-Narrow 62.5 GB/s（Zen 5 桌機 9900X）〔二手〕Chips and Cheese | 16 核≈2 顆 CCD；同節點 8 個 GPU job 共用 | CPU 注意力在 q 小時被頻寬卡住；同時 H2D DMA 也吃這份頻寬 | **有** |
| EPYC 9684X | 3D V-Cache（1,152 MB L3／socket） | 快取 | AMD 產品頁〔文件〕；12 顆 CCD → 每顆 96 MB〔算術〕 | 無 | L3 是每顆 CCD 私有 | KV 是 14–48 GiB、讀一次就走 → **無關**〔判讀〕 | 有（MI300X 主機） |
| Intel DSA（SPR 起） | 搬運引擎 | memcpy 卸載 | — | 單顆飽和 30 GB/s；3、4 顆並用 70、90 GB/s〔全文〕Kuper ASPLOS'24 §4 | 只有 Intel | 機制見 L0；**EPYC 沒有** | 沒有 |
| Intel IAA（SPR 起） | Deflate 壓縮／解壓 | 解壓 | 每 socket 最多 4 顆（L0 已記；LKML 測機亦為 4／socket）〔二手〕 | **單顆 GB/s 本次查不到** | 只有 Intel | 見 L0 | 沒有 |
| CXL 記憶體擴充 | CXL.mem（PCIe 5.0） | 第三層記憶體 | 一個 DDR5-4800 通道 38.4 GB/s | 三顆真實 CXL 裝置只拿到理論頻寬的 46%、47%、20%（All Read）；遠端 NUMA DDR5 70%〔全文〕Sun et al. MICRO'23 §4.2 O4 | 要支援 CXL 的平台 | 一顆約 17.7 GB/s〔算術〕，比 PCIe H2D 慢 → 不是更快的層 | 沒有 |
| NVIDIA H200（Hopper sm_90） | 第 4 代 tensor core（WGMMA） | BF16／FP8 矩陣乘 | BF16 1,979、FP8 3,958 TFLOPS（**含 2:4 稀疏**）→ 稠密約一半〔文件〕NVIDIA H200 規格表 | repo：BF16 matmul **788.1 TFLOPS**（RUNLOG_H200.md:63）；FA3（H100 SXM5）：FP16 740 TFLOPS、FP8 約 1.2 PFLOPS〔全文〕 | FP8 縮放由軟體做（每 tensor 或每塊；硬體沒有區塊縮放，DeepSeek-V3 §3.3） | 縮圖上 prefill 挑塊（算力型）可用 FP8 | **有** |
| H200 | 2:4 稀疏 tensor core | 稀疏矩陣乘 | 規格表的「with sparsity」＝稠密的 2 倍〔文件〕 | 無 H100／H200 的 KV 數字；HieraSparse 在 L40S／A100／4090／A40〔全文〕 | 每 4 個連續元素（沿內積維）恰好 2 個為 0；只能是第 1 個運算元 | 縮圖的 K／V 再砍到約 56% 位元組；壓 K 掉分多 | 有 |
| H200 | TMA | 全域↔共享記憶體的張量搬運 | — | — | 只在 GPU 內部 | 還原 kernel 的實作細節，不改 PCIe 成本〔判讀〕 | 有 |
| H200 | Copy engine（DMA） | H2D／D2H | **官方數量本次查不到** | repo：pinned H2D 53.55 GiB/s、D2H 50.04 GiB/s；同時做 D2H 時 H2D 53.40 GiB/s（RUNLOG_H200.md:63、153） | — | 已是還原的瓶頸（PCIe） | 有 |
| NVIDIA Blackwell（B200／B300） | tcgen05＋TMEM、NVFP4（16 元素一塊、E4M3 縮放）、MXFP4（32 元素、E8M0） | 微縮放低位元矩陣乘 | NVIDIA 部落格〔文件〕 | NVFP4 KV：<1% 準確率損失（NVIDIA 自測）；**目前實作在注意力前先反量化回 FP8**〔文件〕 | 只有 Blackwell | FP4 KV 只省位元組，不省注意力算力 | 沒有 |
| Blackwell | 硬體解壓引擎（DE） | LZ4／Snappy／Deflate 解壓，整合在 copy engine，可「邊傳邊解」 | 最高 600 GB/s〔文件〕nvCOMP DE FAQ | B200 LZ4 解壓輸出 160–220 GB/s（64 KB 塊）〔全文〕arXiv 2512.02189 Table II | 只有 B200／B300／GB200／GB300；單塊上限 4 MiB；要特殊配置的 pinned 記憶體 | 無損壓縮的 CPU 層可邊傳邊解；**我們沒有** | 沒有 |
| GPU 軟體解壓（Hopper） | nvCOMP（ANS、Bitcomp、LZ4…）、DietGPU、DFloat11 | 在 SM 上解壓 | — | H100：DietGPU ANS 解碼 592.5 GB/s、nvCOMP ANS 端到端 480 GB/s〔部分，單一作者預印本〕；DFloat11 比 nvCOMP ANS 快到 20.97×〔全文，絕對值只在圖上〕 | 佔 SM | 可能翻掉「無損只在慢層划算」（§4） | H200 可裝（未裝） |
| AMD MI300X（CDNA3 gfx942） | MFMA、FP8 | BF16／FP8 矩陣乘 | BF16 1,307.4、FP8 2,614.9 TFLOPS（稀疏 ×2）；HBM 5.3 TB/s〔文件〕CDNA3 白皮書 p.24 | 本次沒查 MI300X 的注意力實測 | FP8 是 **FNUZ** 格式（ROCm 文件），和 H100 的 OCP FP8 不同 | 同 H200 | **有** |
| MI300X | SMFMAC（稀疏 MFMA） | 2:4 稀疏矩陣乘 | 每 CU 每時脈最多 8K 運算〔文件〕白皮書 p.8 | 無 KV／注意力實測 | 每 4 個沿 K 維的元素至少 2 個為 0；一個運算元稀疏、另一個稠密；hipSPARSELt 支援 FP16、BF16、INT8、FP8 FNUZ〔文件〕 | 同 H200 的 2:4；沒有現成注意力 kernel | 有 |
| MI300X | SDMA 引擎 | H2D／D2H | ConCCL 說 14 個（repo L2 已記）；AMD 自己的 DMA-Latte 說 16 個〔部分〕 | vLLM 0.28 CPU→GPU 4.1 GiB/s（軟體逐塊 copy，pinned 約 50 GiB/s）（L7c） | 預設 `HSA_ENABLE_SDMA=1`（L0） | 瓶頸是軟體，不是引擎數 | 有 |
| AMD MI355X（CDNA4 gfx950） | MXFP8／MXFP6／MXFP4 微縮放 MFMA | 低位元矩陣乘 | MXFP6／MXFP4 10 PF（MI355X）；FP8 改為 OCP；MXFP4／6 沒有稀疏版〔文件〕CDNA4 白皮書 p.7–8、p.18 | 無 | 只有 CDNA4 | 無 | 沒有 |
| CDNA3／CDNA4 | 硬體解壓 | — | 兩份白皮書全文都沒有 decompress 一詞〔全文〕 | — | — | 只能用 hipCOMP／hipANS 軟體（L0） | 沒有 |

---

## §2 卡片

### 2.1 CPU 端

#### F1：Intel AMX 的峰值怎麼來（Intel® 64 and IA-32 Architectures Optimization Reference Manual Vol. 1，文件號 248966-051，2026-10）
- **讀了什麼**：〔全文〕§20.2.1、§20.3 Table 20-2（PDF 第 673 頁，文件頁 20-3）、§20.5.2（文件頁 20-4）、§2.1.1（PDF 第 46 頁）。<https://cdrdv2.intel.com/v1/dl/getContent/671488>，查證 2026-10-27。
- **一句話**：AMX 是 CPU 核心裡的小矩陣乘單元，一條指令乘兩塊 tile。
- **關鍵事實**：
  - Table 20-2：`TDP/*` 吞吐 16 週期、延遲 52；`TILELOADD` 吞吐 8；`LDTILECFG` 延遲 204〔文件〕。
  - §20.5.2：A tile 最多 16 列 × MAX_TILE_K 欄，MAX_TILE_K＝64／sizeof(type)，BF16 為 32；C tile 最多 16×16 FP32〔文件〕。
  - 〔算術〕一條最大 BF16 `TDPBF16PS`＝16×16×32＝8,192 次乘加＝16,384 FLOP；÷16 週期＝**1,024 FLOP／週期／核**。和 Intel 架構日「1024 bfloat16 operations per cycle」〔二手〕一致；KTransformers 的「8452Y 理論 73.7 TFLOPS」＝36 核×1,024×2.0 GHz，也一致。
  - §20.2.1：AMX 有自己的頻率等級，「usually lower」than AVX-512 等級；利用率低於 15% 時可超過〔文件〕。**實際 AMX 時脈沒有數字**〔未查證〕。
  - §2.1.1：「AMX supports FP16 … (Granite Rapids only)」〔文件〕。
- **成立的前提**：M 方向最多 16 列。注意力每個 KV head 的 M＝問題長度 q × GQA 組大小 G（Qwen-7B 的 G＝7），q＝1 時只用到 7／16 列〔算術〕。
- **跟 Tiara 的關係**：見 §3。我們的平台沒有 AMX。
- **與既有整理的關係**：`arith.md` §7 寫「沒有 AMX」，與 `results/hw_h200.json`（EPYC 9554）一致。

#### F2：Intel 未來的 AMX 擴充（Intel Architecture Instruction Set Extensions and Future Features Programming Reference，319433-062，2026-06）
- **讀了什麼**：〔部分〕修訂紀錄與 Table 1-2（PDF 第 16–17 頁）。<https://cdrdv2-public.intel.com/922690/319433-062-architecture-instruction-set-extensions-programming-reference.pdf>，查證 2026-10-27。
- **事實**：Table 1-2：AMX-FP16＝Granite Rapids；AMX-COMPLEX＝Granite Rapids D（06_AEH）、Diamond Rapids；AMX-MOVRS、AMX-AVX512、AMX-FP8＝Diamond Rapids〔文件〕。修訂紀錄寫這一版從 Table 1-2 刪掉 AMX-TF32，另一筆刪掉 AMX-TRANSPOSE〔文件〕。
- **跟 Tiara 的關係**：AMX-FP8 在 CPU 端理論上可直接吃 FP8 KV，但沒有吞吐數字、沒出貨 → 無關〔判讀〕。

#### F3：Zen 4／Zen 5 的 AVX-512 BF16 吞吐（uops.info）
- **讀了什麼**：〔全文〕<https://uops.info/html-instr/VDPBF16PS_ZMM_ZMM_ZMM.html>、<https://uops.info/html-instr/VFMADD231PS_ZMM_ZMM_ZMM.html>，查證 2026-10-27。
- **量到的數字**：`VDPBF16PS (ZMM)`：Zen 4 實測吞吐 1.00 週期／條（延遲 6），AMD 文件欄也寫 1.00；Zen 5 實測 0.50；Emerald Rapids 2.00。`VFMADD231PS (ZMM)`：Zen 4 1.00、Zen 5 0.50〔全文〕。
- **算術**：一條 zmm `VDPBF16PS`＝16 條 FP32 通道，各做 2 乘 2 加＝64 FLOP。Zen 4：**64 FLOP／週期／核**；Zen 5：128〔算術〕。Zen 4 的 512-bit 指令拿到的速度與兩條 256-bit 相同（FMA zmm 1／週期），符合「拆兩半」的描述〔判讀〕。
- **Zen 5 伺服器是不是真 512-bit**：AMD 2024-10-10 新聞稿列「AVX-512 with the full 512b data path」〔文件〕<https://ir.amd.com/news-events/press-releases/detail/1219/amd-launches-5th-gen-amd-epyc-cpus-maintaining-leadership>。uops.info 的 Zen 5 量測機型號頁面沒寫〔未查證〕。Zen 5c、行動版的寬度本次沒查〔未查證〕。
- **跟 Tiara 的關係**：R2 的 3.2 TFLOPS＝16×64×3.1 GHz，**每週期 64 這個係數有出處**；剩下的不確定只在時脈（3.1 基頻 vs 3.75 全核加速，AMD 產品頁）。
- **與既有整理的關係**：L0 §3.1 記 MI300X 主機 `/proc/cpuinfo` 有 AVX-512 F／BF16／VNNI、沒有 AMX，一致。

#### F4：EPYC 9554 的記憶體頻寬（AMD 產品頁；Fujitsu PRIMERGY RX1440 M2 Performance Report v1.9）
- **讀了什麼**：AMD 產品頁〔文件〕<https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9554.html>；Fujitsu 報告〔全文〕STREAM 章（PDF p.18–20）<https://sp.ts.fujitsu.com/dmsp/Publications/public/wp-performance-report-primergy-rx1440-m2-ww-en.pdf>，查證 2026-10-27。
- **事實**：9554＝64 核、基頻 3.1 GHz、全核加速 3.75 GHz、L3 256 MB、12 通道 DDR5-4800、460.8 GB/s／socket〔文件〕。Fujitsu 記憶體白皮書 v1.3 p.6：第 4 代 EPYC 的 CCX「up to 8 processor cores and up to 96 MB of L3」〔全文〕→ 64 核至少 8 個 CCX；以每顆 CCD 8 核、L3 256÷8＝32 MB 解讀為 8 顆 CCD〔算術＋判讀；AMD 官方的 9554 CCD 數本次沒讀到〕。
- **量到的數字**：1×EPYC 9554P、12×32 GB 2Rx8 DDR5-4800、SMT 關、NPS4、RHEL 9.0：**STREAM Triad 390 GB/s**（p.20 表）〔全文〕。同表 4 顆 CCD 的型號（9124、9224）只有 257–266 GB/s，8 顆 CCD 以上的型號都在 360–398〔全文〕→ 少量 CCD 時，被 CCD 到 IOD 的連線卡住〔判讀〕。
- **每顆 CCD 的頻寬**：Chips and Cheese 量 Zen 5：GMI-Wide 單 CCD 讀 99.8 GB/s，桌機 9900X 的 GMI-Narrow 單 CCD 讀 62.5 GB/s〔二手，WebFetch 轉述〕<https://chipsandcheese.com/p/amds-epyc-9355p-inside-a-32-core>。Genoa（Zen 4）每顆 CCD 的官方或實測值**本次查不到**〔未查證〕（AMD 58015 架構文件下載被擋成 HTML）。
- **我們 job 拿得到多少**：job 的 cpuset 是連續 16 核（80–95 或 96–111，`H200_SLURM.md`、`results/hw_h200.json`）→ 若 Linux 核心編號按 CCD 連續，就是 2 顆 CCD〔判讀，未驗證編號規則〕。兩條估法：(a) 2×62.5＝125 GB/s；(b) 同一個 socket 有 4 個 GPU job（節點 8 卡、`cotenant_jobs_on_node: 7`），平分 390 → 98 GB/s〔算術〕。§3 用 98–125 GB/s。
- **跟 Tiara 的關係**：pinned H2D DMA 也是從同一份 DRAM 讀；別的 job 在搬 KV 時，CPU 注意力和 PCIe 搬運搶同一份頻寬〔判讀〕。

#### F5：EPYC 9684X 的 3D V-Cache（AMD 產品頁）
- **讀了什麼**：〔文件〕<https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9684x.html>：96 核、基頻 2.55 GHz、L3 1,152 MB、460.8 GB/s。
- **判讀**：Fujitsu 白皮書 p.6 寫每個 CCX 最多 96 MB L3〔全文〕；1,152 MB÷96 MB＝12 顆 CCD、每顆 8 核〔算術〕，而且 L3 是每個 CCX 私有。KV（Qwen-7B 256K 14 GiB）讀一次就走，連縮圖（1/16 也要 0.9 GiB）都放不進單顆 CCD → **對還原無關**。

#### F6：Intel DSA／IAA 的新數字（只補 L0 沒有的）
- **DSA**：〔全文〕Kuper et al., *A Quantitative Analysis and Guidelines of Data Streaming Accelerator in Modern Intel Xeon Scalable Processors*（ASPLOS'24；arXiv 2305.02480v5 HTML）。單顆 DSA 吞吐飽和在 30 GB/s，原文歸因於 I/O fabric 上限；3、4 顆一起用，64 KB 以上最高 70、90 GB/s（Fig. 4、Fig. 10 的內文）。<https://arxiv.org/html/2305.02480v5>
- **IAA**：每 socket 最多 4 顆、整機最多 8 顆（LKML zswap 補丁的測機：SPR 2 socket、每 socket 4 顆）〔二手〕。**單顆 Deflate 解壓 GB/s 本次查不到**（查詢：`"IAA" deflate decompression "GB/s" QPL benchmark single device`）。
- **AMD 端的對應物**：L0 §3.2 已記 PTDMA／AE4DMA（為 NTB 設計），本次沒有找到 EPYC 上的壓縮加速器（查詢：`AMD EPYC on-die compression accelerator`，只做到 CDNA 白皮書全文檢索）〔未查證〕。

#### F7：CXL 記憶體當一層（Sun et al., *Demystifying CXL Memory with Genuine CXL-Ready Systems and Devices*，MICRO'23，arXiv 2303.15375）
- **讀了什麼**：〔部分〕§2–§4.2（pdftotext），<https://arxiv.org/pdf/2303.15375>，查證 2026-10-27。
- **量到的數字**：2×Xeon 6430（SPR）＋三顆 CXL 記憶體（CXL-A：硬 IP＋DDR5-4800；CXL-B：硬 IP＋2×DDR4-2400；CXL-C：FPGA 軟 IP＋DDR4-3200）。「All Read」只拿到理論頻寬的 46%、47%、20%；遠端 NUMA 的 DDR5 是 70%（§4.2 O4）〔全文〕。
- **算術**：CXL-A 0.46×38.4＝17.7 GB/s〔算術〕，比 H200 pinned H2D（57.5 GB/s）慢。
- **跟 Tiara 的關係**：在我們的動作空間裡，它是「容量」層，不會讓還原比 CPU 層快〔判讀〕。

#### F8：CPU 端注意力／專家卸載論文的硬體事實（只記硬體，演算法由 agent C 負責）
- **KTransformers**（SOSP'25，Chen et al.；PDF <https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf>）〔全文〕：
  - 2×Xeon Platinum 8452Y（各 36 核）、各 1 TB DDR5；Intel MLC 量到 socket 內 220 GB/s、跨 socket 125 GB/s；A100-40GB 與 RTX 4080 都走 PCIe 4.0（§6.1，PDF p.10）。
  - PyTorch AMX MoE 最高 5.4 TFLOPS、AVX-512 1.8，約為 AMX 理論 73.7 TFLOPS 的 7%（§2，p.4）。
  - 自己的 AMX kernel 單 socket 21.3 TFLOPS（§3.2，p.6）。
  - 「每個專家 ≤4 個 token」時 AVX-512 一直贏 AMX（§3.2，Fig. 7）。
- **NEO**（arXiv 2411.01142，MLSys'25）〔部分〕：
  - 平台：g5（A10G＋EPYC 7R32）、g4（T4＋Xeon 8259CL）、HGX 8×H100＋Xeon 8462Y+（64 核）、2 TB（Table，p.7）。
  - CPU 注意力用 ISPC 編譯成 AVX2／AVX-512（p.5）。
  - 敏感度實驗的結論：吞吐增益跟 CPU 記憶體頻寬正相關，不跟核數正相關（§5.5）。
  - 它的動機數字「GPU 600 vs CPU 200 GB/s」（p.3）是 A10G 對一般 x86 伺服器。
- **LIA**（ISCA'25，Kim et al.）〔摘要〕：AMX 矩陣乘在 SPR 20 TFLOPS、GNR 40 TFLOPS。型號、核數、時脈本次沒讀到。<https://deepmind.google/research/publications/81986/>（全文 <https://dl.acm.org/doi/full/10.1145/3695053.3731092>，未讀）。

### 2.2 NVIDIA

#### F9：H200 規格（NVIDIA 產品頁）
- **讀了什麼**：〔文件〕<https://www.nvidia.com/en-us/data-center/h200/> 規格表，查證 2026-10-27。
- **事實**：H200 SXM：TF32 989、BF16／FP16 1,979、FP8／INT8 3,958 TFLOPS，註 2「With sparsity」；HBM 141 GB、4.8 TB/s；PCIe Gen5 128 GB/s；NVLink 900 GB/s〔文件〕。所以稠密 BF16 約 989、FP8 約 1,979〔算術〕。
- **對照實測**：repo 在 H200 上 BF16 matmul 788.1 TFLOPS（稠密峰值的 80%）、HBM 裝置內複製 4,173.9 GB/s、pinned H2D 53.55 GiB/s（`results/RUNLOG_H200.md:63`）。

#### F10：Hopper 的 FP8、TMA、2:4 稀疏（NVIDIA *Hopper Architecture In-Depth* 部落格；cuSPARSELt 文件）
- **讀了什麼**：〔部分〕<https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/>（TMA 段、FP8 段、Transformer Engine 段）；cuSPARSELt 首頁 <https://docs.nvidia.com/cuda/cusparselt/index.html>，查證 2026-10-27。
- **事實**：
  - FP8 有 E4M3、E5M2 兩種〔文件〕。
  - Transformer Engine 用「從 tensor 統計算出的縮放因子」把資料放進可表示範圍，也就是每個 tensor 一個縮放，由軟體算〔文件〕。
  - TMA 用 copy descriptor 在全域記憶體與共享記憶體之間搬 1D–5D 張量，單一執行緒就能發〔文件〕。
  - 稀疏：cuSPARSELt「至少一個運算元是 50% 稀疏的結構化矩陣」，支援 SM 8.0 到 12.1（含 9.0）〔文件〕。
- **Hopper 沒有硬體區塊縮放**：DeepSeek-V3 技術報告 §3.3（arXiv 2412.19437v2）說 H800 上 FP8 GEMM 的累加精度只保留約 14 bits。它的作法是用軟體做 1×128／128×128 區塊縮放，每 128 個元素提升到 CUDA core 用 FP32 累加；並指出 Blackwell「announced the support for microscaling formats」〔全文〕。
- **FA3 實測**（Shah et al., arXiv 2407.08608v2）〔部分，摘要與 §1、§4 內文〕：H100 SXM5 上 FP16 前向最高 740 TFLOPS（75% 利用率）、FP8 接近 1.2 PFLOPS。用區塊量化＋incoherent processing 後，FP8 誤差比「每 tensor 量化的 FP8 注意力」低 2.6 倍。
- **跟 Tiara 的關係**：見 §4。

#### F11：HieraSparse：*Hierarchical Semi-Structured Sparse KV Attention*（arXiv 2604.16864v2，2026-08-17）
- **讀了什麼**：〔部分〕HTML 摘要、§II、§III-C、§V-A、跨 GPU 段，<https://arxiv.org/html/2604.16864>，查證 2026-10-27。
- **一句話**：把 K 與 Vᵀ 剪成 2:4，當稀疏 tensor core 的第一運算元，加速 prefill 與 decode 的注意力。
- **硬體的關鍵一步**：稀疏 tensor core 只壓第一個運算元，所以要把兩個 GEMM 都轉置（算 Sᵀ、Oᵀ）（§III-C，Table II「Trans-Both」）。FP16 2:4 的 metadata 是稠密大小的 1/16（§III-D）→ 位元組約 0.5＋0.0625＝0.5625 倍〔算術〕。
- **成立的前提**：不需訓練；用 magnitude pruning；sink、heavy hitter、local window 留稠密（§I）。
- **量到的數字**：
  - 主平台 L40S。跨 GPU：A100 prefill 1.45×／decode 1.74×、RTX 4090 1.80×／1.72×、A40 1.79×／1.81×（都是注意力 kernel 對 FP16 稠密）〔全文〕。
  - **沒有 Hopper**：kernel 用 `mma.sp`，不是 `wgmma.sp`〔全文；判讀：在 H200 上會跑，但不會用到 Hopper 的 WGMMA 快路徑〕。
  - 品質（Table IV，Llama-3.1-8B）：prefill 只壓 V、decode 也只壓 V 時 47.32 分，注意力 1.34×／1.28×；decode 連 K 也壓時 45.59 分，1.34×／1.71×〔全文〕。
  - 只在 prefill 壓 K：LongBench 掉 11.1；只壓 V：掉 1.8（§V-A，Fig. 6a）〔全文〕。
- **跟 Tiara 的關係**：見 §4（縮圖可以是 2:4 的 V；K 不行）。
- **與既有整理的關係**：對 repo 的 docs／results 跑 `grep -rniE "HieraSparse|2:4"`，只比到時間戳記，沒有這篇〔判讀：是新的〕。

#### F12：Blackwell 的微縮放格式（NVIDIA NVFP4 兩篇部落格）
- **讀了什麼**：〔部分〕*Introducing NVFP4…*（2025-06-24，WebFetch 轉述）<https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>；〔全文〕*Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache*（2025-12-08）<https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/>，查證 2026-10-27。
- **事實**：
  - NVFP4＝E2M1，每 16 個值一個 E4M3 縮放，外加每 tensor 一個 FP32；MXFP4＝每 32 個值一個 E8M0（2 的冪）縮放〔文件〕。
  - NVFP4 KV 比 FP8 KV 省約 50% 記憶體；在 LiveCodeBench、MMLU-PRO、MBPP、Ruler 64K 上準確率損失 <1%；Llama 3.3 70B 的 MMLU，NVFP4 KV 比 MXFP4 KV 高 5%〔文件，NVIDIA 自測〕。
  - **「current implementation … dequantized from NVFP4 to FP8 before attention」**〔文件〕。
- **跟 Tiara 的關係**：就算到 Blackwell，FP4 KV 在注意力裡也還是先變回 FP8 → 省的是位元組，不是注意力算力〔判讀〕。我們沒有 Blackwell。

#### F13：Blackwell 硬體解壓引擎（nvCOMP DE FAQ；NVIDIA 部落格；Blackwell 微基準 arXiv 2512.02189v3）
- **讀了什麼**：〔全文〕DE FAQ <https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html>；〔全文〕部落格 <https://developer.nvidia.com/blog/speeding-up-data-decompression-with-nvcomp-and-the-nvidia-blackwell-decompression-engine/>；〔部分〕微基準 §IV-A2、§V-B、Table I–II，<https://arxiv.org/html/2512.02189v3>，查證 2026-10-27。
- **事實**：
  - 「up to 600 GB/s」；固定功能，支援 Snappy、LZ4、Deflate、Gzip。
  - 與 copy engine 整合：壓縮資料可直接走 PCIe 或 C2C，邊傳邊解。
  - 只在 B200、B300、GB200、GB300。
  - B200 單塊上限 4 MiB，超過就退回 SM。
  - H2D 解壓要用 `cudaMemPoolCreateUsageHwDecompress` 或 `CU_MEM_CREATE_USAGE_HW_DECOMPRESS` 配置 pinned 記憶體〔文件〕。
- **量到的數字**：B200、100 MB、64 KB 塊，LZ4 解壓輸出 160–220 GB/s（亂數資料 172.55、重複資料 219.80）〔全文〕Table II。
- **與文件不一致**：該論文 Table I 也列 zstd 154.9、Bitcomp 462.4、ANS 539.2 GB/s 為「DE」結果。但 NVIDIA 文件說 DE 只支援 LZ4／Snappy／Deflate；這幾列應是 SM 上的 nvCOMP，論文沒有分開〔判讀〕。
- **跟 Tiara 的關係**：理論上正好對到「無損壓縮的 CPU 層，GPU 邊傳邊解」。但 (a) 我們沒有 Blackwell；(b) LZ4／Deflate 對 BF16 KV 的壓縮比沒量過（repo 的 1.47 是 zstd:1）。

#### F14：Hopper 上 GPU 軟體解壓的實測
- **DietGPU／nvCOMP ANS on H100**〔部分〕：Shavidze, *Compressed-Resident Genomics…*，arXiv 2606.18900v1（單一作者、非同儕審查），§6.4（PDF p.4）：H100 80GB，256 MB 指數分布資料、4096×64 KB 批次，DietGPU ANS 編碼 364.9 GB/s、解碼 592.5 GB/s、逐位元正確；作者自己端到端量的 nvCOMP ANS 是 480 GB/s。<https://arxiv.org/pdf/2606.18900>。**單一來源，可信度低**。
- **DFloat11**〔部分〕（arXiv 2504.11651v3 §5）：BF16 權重的指數用 Huffman 編碼，壓到約 70%（文中 68%，nvCOMP ANS 79%）。用 Llama-3.1-8B LM head 的切片比較，DF11 解壓吞吐最高是 CPU→GPU 搬運的 34.95×、nvCOMP ANS 的 20.97×。**絕對 GB/s 只在 Fig. 7 圖上**，文字沒有；GPU 型號為 A100／H100 等多種〔全文〕。
- **nvCOMP 官方**：文件只有演算法選擇建議，沒有 Hopper 的數字（algorithm_selection 頁）〔文件〕。
- **跟 Tiara 的關係**：見 §4.1。這是「無損壓縮在 CPU 層」唯一可能翻盤的路。
- **與既有整理的關係**：L0 E-L1 已提「GPU 端無損 BF16 指數編碼」，預測 CPU 層「省 ≤5% 或變慢」。前提是解碼 ≥28 GiB/s（SplitZip）。依本卡的數量級（數百 GB/s），這個預測要重算（§4.1）。E-L1 沒有在 repo 跑過（`experiment_log.md` 只有 S1 的 CPU 解壓）。

#### F15：H200 的 copy engine 數量
- **本次查不到**（查詢：`H100 deviceQuery "copy engine(s)"`、`"H100" "copy engines" number`）。NVIDIA 論壇有人說官方文件沒有逐型號列出〔二手〕。
- **repo 已有的間接證據**：C0 量到單獨 H2D 53.53 GiB/s，同時做 D2H 時 H2D 53.40（`results/RUNLOG_H200.md:153`）→ 兩方向至少各有一個獨立引擎〔判讀〕。
- 要確定的話，在 H200 上跑 `deviceQuery` 或讀 `cudaDeviceProp.asyncEngineCount` 就有（不能在 Mac 上做）。

### 2.3 AMD GPU

#### F16：MI300X（CDNA3）的 MFMA、FP8、稀疏（AMD CDNA 3 白皮書；ROCm precision support；hipSPARSELt MI300 文件）
- **讀了什麼**：〔全文〕CDNA3 白皮書（pdftotext），<https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf>；〔部分〕<https://rocm.docs.amd.com/en/latest/reference/precision-support.html>；〔部分〕<https://rocm.docs.amd.com/projects/hipSPARSELt/en/latest/conceptual/mi300-features.html>，查證 2026-10-27。
- **事實**：
  - MI300X：BF16 1,307.4／2,614.9（稀疏）、FP8 2,614.9／5,229.8（稀疏）TFLOPS；HBM 5.3 TB/s；PCIe Gen5 x16 到主機（白皮書 p.24 表）〔文件〕。
  - 稀疏條件（p.8）：「每 4 個輸入值裡至少 2 個是 0」，非零值壓成緊湊格式＋metadata，每 CU 每時脈最多 8K 運算〔文件〕。
  - hipSPARSELt：用 SMFMA 指令；「Two out of every four elements on the K axis must be zero」；一個運算元稀疏、另一個稠密。目前硬體加速的型別是 FP16、BF16、INT8、FP8 FNUZ、BF8 FNUZ〔文件〕。
- **FP8 格式的不一致**：CDNA3 白皮書 p.8 寫 CDNA3 支援「OCP 8-bit Floating Point Specification」的 E5M2／E4M3。ROCm precision support 頁則寫「CDNA3 natively supports FP8 FNUZ…, which differs from … NVIDIA H100」，型別表也只在 CDNA3 打勾 `__hip_fp8_e4m3_fnuz`，OCP 的 `__hip_fp8_e4m3` 在 CDNA4 才打勾〔文件〕。**以 ROCm 文件為準**（較具體，且與 hipSPARSELt 的 FNUZ 一致）〔判讀〕。
- **跟 Tiara 的關係**：H200 上做的 FP8 縮圖，搬到 MI300X 要換成 FNUZ（範圍、NaN 表示不同），不能直接逐位元比較〔判讀〕。2:4 稀疏的條件與 NVIDIA 相同，但本次沒找到 ROCm 上用稀疏 MFMA 做注意力的 kernel（查詢：`MI300 sparse attention SMFMAC KV`）。

#### F17：MI300X 的 SDMA 引擎
- repo 已記：ConCCL 說 IOD 上有 14 個 SDMA（`docs/stage3_20261019/lit/L2_compression_offload_prefetch.md:290、377`）。
- **新的、不一致的來源**：AMD 自己的 DMA-Latte（arXiv 2511.06605，Pati et al.，§III-B，Fig. 4）寫每顆 MI300X「16 system-DMA (sDMA) engines」〔部分〕。兩者差 2 個，原因不明〔未查證〕。
- 預設是否走 SDMA：`HSA_ENABLE_SDMA` 預設 1（L0 §3.2，A8）。vLLM 只拿到 4.1 GiB/s 是逐塊小 copy 的固定成本，不是引擎不夠（L7c §2.1、`experiment_matrix.md:158`）。

#### F18：MI355X（CDNA4）的微縮放（AMD CDNA 4 白皮書）
- **讀了什麼**：〔部分〕p.6–8、p.18，<https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf>，查證 2026-10-27。
- **事實**：
  - 支援 OCP MX 的 MXFP8、MXFP6（E3M2／E2M3）、MXFP4（E2M1），「block n × 32」共用 8-bit 指數（Figure 4）。
  - MI355X：MXFP6／MXFP4 10 PF、OCP-FP8 5.0／10 PF（稀疏）、BF16 2.5／5.0 PF。
  - **MXFP6／MXFP4 那兩列沒有稀疏版**。
  - TF32 改成軟體模擬（p.7–8、p.18）〔文件〕。
- **跟 Tiara 的關係**：沒有這張卡；沒有證據顯示 MXFP4 在 AMD 上能直接當注意力的運算元而不先反量化〔未查證〕。

### 2.4 3090 伺服器
- CPU 型號：repo 只記「96 核、440 GB、PCIe gen3」（`docs/ENV_3090.md:165`）。**型號本次查不到**（repo 無 lscpu 記錄）〔未查證〕。

---

## §3 R2 重算：在 CPU 上直接算注意力，什麼時候贏過把 KV 搬上 GPU

**一句話**：照實際硬體重算，R2 在 H200 節點（EPYC 9554、16 核）上**成立**：打平點 8–17 個問題 token，而我們的問題是 128–512 個。換成 AMX Xeon 的 16 核，打平點最多 23–41（用實測換算）；整顆 GNR 也只到 99–174。**只有在 MI300X＋vLLM 0.28（4.1 GiB/s 的軟體缺陷）上才會翻盤**，而那是工程問題。

### 3.1 公式〔算術〕

記號：L＝層數、h_kv＝KV head 數、h_q＝query head 數、d＝head 維度（128）、G＝h_q／h_kv、n＝快取 token 數、q＝問題 token 數。

- KV 位元組（BF16）：B＝2·L·h_kv·d·2·n
- 注意力 FLOPs（QKᵀ 與 PV 各 2 FLOP／乘加；softmax 不計）：F(q)＝4·q·n·h_q·d·L
- **算術強度** I＝F／B＝q·h_q／h_kv＝**q·G FLOP／byte**。KV 每個位元組只讀一次，所有 query head 與問題 token 共用。
- 搬 KV 過 PCIe：t_load＝B／B_PCIe，B_PCIe＝53.55 GiB/s（H200 實測）
- CPU 注意力（roofline，樂觀）：t_cpu(q)＝max(B／B_DRAM, F(q)／P_cpu)
- 打平點：t_cpu(q\*)＝t_load。當 B／B_DRAM < t_load 時，q\*＝t_load／(F(1)／P_cpu)。

| 模型・長度 | L, h_kv, h_q | B (GB) | F(1) | G | t_load (s) |
|:--|:--|--:|--:|--:|--:|
| Qwen2.5-7B-1M・256K | 28, 4, 28 | 15.03 | 1.052e11 | 7 | 0.261 |
| Qwen2.5-14B-1M・256K | 48, 8, 40 | 51.54 | 2.577e11 | 5 | 0.896 |
| Llama-3.1-8B・128K | 32, 8, 32 | 17.18 | 6.872e10 | 4 | 0.299 |

### 3.2 峰值的出處

| 平台（每個 GPU job 能用的部分） | P_cpu | 出處 | B_DRAM |
|:--|--:|:--|:--|
| EPYC 9554，16 核，3.1 GHz 基頻 | 16×64×3.1e9＝**3.17 TFLOPS** | 64 FLOP／週期：uops.info Zen 4 `VDPBF16PS zmm` 吞吐 1〔全文〕；3.1 GHz：AMD 產品頁〔文件〕 | 98–125 GB/s（F4 兩種估法）〔算術＋二手〕 |
| 同上，3.75 GHz 全核加速 | 3.84 TFLOPS | AMD 產品頁 All Core Boost 3.75 GHz〔文件〕 | 同上 |
| EPYC 9554 整顆 64 核（假設獨佔） | 12.70 TFLOPS | 同上 | 390 GB/s（Fujitsu STREAM）〔全文〕 |
| EPYC 9684X，32 核（MI300X 容器配額），2.55 GHz | 5.22 TFLOPS | 同上＋9684X 產品頁〔文件〕 | 本次沒算（4 顆 CCD 推估約 200–250 GB/s〔判讀〕） |
| Intel AMX，16 核，峰值 @2.0 GHz | 16×1,024×2.0e9＝32.8 TFLOPS | Intel Opt. Manual Table 20-2、§20.5.2〔文件＋算術〕；2.0 GHz 是代用值，AMX 時脈〔未查證〕 | SPR 每 socket 理論 8×38.4＝307 GB/s〔算術〕；KTransformers MLC 實測 220 GB/s〔全文〕 |
| Intel AMX，16 核，**實測換算** | 21.3／36×16＝9.47 TFLOPS | KTransformers 單 socket 21.3 TFLOPS（MoE GEMM，高強度）〔全文〕 | 同上 |
| AMX 整顆 SPR／GNR（實測） | 20／40 TFLOPS | LIA〔摘要〕 | 本次沒查 GNR 實測頻寬〔未查證〕 |

### 3.3 結果〔算術〕

每個問題 token 的 CPU 注意力時間 t₁＝F(1)／P_cpu；打平點 q\*＝t_load／t₁（記憶體頻寬那一項先檢查過，都小於 t_load）。

| 平台 | Qwen-7B 256K：t₁ (ms) → q\* | Qwen-14B 256K：t₁ (ms) → q\* | Llama-8B 128K：t₁ (ms) → q\* |
|:--|:--|:--|:--|
| 9554×16 核 @3.1（＝repo 假設） | 33.1 → **7.9** | 81.2 → **11.0** | 21.6 → 13.8 |
| 9554×16 核 @3.75 | 27.4 → 9.5 | 67.1 → 13.4 | 17.9 → 16.7 |
| 9554×64 核（獨佔整顆） | 8.3 → 31.5 | 20.3 → 44.2 | 5.4 → 55.2 |
| Zen 5 ×16 核 @3.1（同時脈假設） | 16.6 → 15.8 | 40.6 → 22.1 | 10.8 → 27.6 |
| AMX ×16 核，實測換算 | 11.1 → 23.5 | 27.2 → 32.9 | 7.3 → 41.2 |
| AMX ×16 核，峰值 | 3.2 → 81.4 | 7.9 → 114.0 | 2.1 → 142.5 |
| AMX 整顆 GNR（LIA 40 TF） | 2.6 → 99.4 | 6.4 → 139.1 | 1.7 → 173.9 |

- **重現 repo 的數字**：第一列 7.9→7、11.0→11 與 `arith.md` §7 的「7、11 token」相同。**R2 的峰值假設有出處，不是猜的。**
- **記憶體頻寬那一項**：9554×16 核在 98–125 GB/s 時，t_mem＝0.12–0.15 s（7B）、0.41–0.53 s（14B），都小於 t_load。roofline 的轉折點在 q≈P／(G·B_DRAM)：7B 是 4.5、14B 是 6.3（@100 GB/s）。所以 **q≤4 時 CPU 被頻寬卡住，而且這時 CPU 注意力確實比搬 KV 快**（0.12–0.15 s 對 0.26 s）。q 再大就變成算力型，然後在 q\* 輸掉。
- **MI300X**：用 vLLM 0.28 的 4.1 GiB/s 算，t_load＝3.42 s → 32 核 9684X 的 q\*≈170〔算術〕，**R2 在這條路徑上會翻盤**。換成 pinned 路徑（約 50 GiB/s，`experiment_matrix.md:158`），t_load＝0.28 s → q\*≈14。所以翻盤只是軟體缺陷造成的（L7c 已列為工程問題）〔判讀〕。
- **R2 沒算到的（強化 R2）**〔判讀〕：
  - 回答時每生成一個 token，都要在 CPU 上把全部 KV 再讀一次（q＝1，頻寬卡住），每個 token 0.12–0.15 s（7B），而 GPU 上 decode 一步的下界是 7.3 ms（arith §8）。除非背景把 KV 搬上 GPU（NEO、HGCA 的作法），否則 decode 階段 CPU 版更慢。
  - 逐層要等 CPU 結果，GPU 會閒置。agent C 文獻裡 HGCA 的 GPU 閒置 57% 是二手數字（本次搜尋摘要），本卡不採用。
  - 同節點 8 個 GPU job 共用兩個 socket 的 DRAM，別人搬 KV 時頻寬更少。
- **什麼條件下 R2 會翻**〔算術〕：問題 ≤ q\*。用 9554 時是「問題 ≤8–17 個 token，而且不生成或只生成很少 token」，例如是非題或分類，不是我們的工作負載。

---

## §4 判讀：哪些硬體功能可能翻掉之前的決定，哪些無關

**一句話**：會改變還原成本的，都在 GPU 那一邊，而且都不是「矩陣乘指令」本身，而是：(1) GPU 端無損解壓、(2) 縮圖上「問題 prefill」這一步用 FP8 tensor core、(3) 用 2:4 稀疏或微縮放格式讓縮圖更小（省的是位元組）。

### 4.1 可能翻掉既有判定：GPU 端無損解壓（S1 的「CPU 層變慢」）
- **既有判定**：S1 量到 BF16 KV 用 zstd:1 可壓 1.47–1.52 倍，但 CPU 解壓只有 0.5–2.1 GB/s → CPU 層大幅變慢，無損只在慢層划算（`experiment_log.md` S1 節；`results_analysis.md` §4.5）。
- **算術**（壓縮比 r＝1.47，PCIe B＝53.55 GiB/s，GPU 解壓輸出速度 D；時間以「BF16 原樣搬」為 1）：
  - 重疊：max(1/r, B/D)；不重疊：1/r＋B/D。
  - D＝30 GB/s：1.91／2.59（變慢）。
  - D＝100 GB/s：0.68／1.26。
  - D＝480 GB/s：0.68／0.80。
  - D＝592 GB/s：0.68／0.78。
- **所以**：只要 GPU 解壓有幾百 GB/s（F14 的 H100 數量級），**CPU 層也能省 20–32%**。這和 L0 E-L1 的事先預測（CPU 層 ≤5% 或變慢，前提 D≈28 GiB/s）不同。
- **條件與風險**〔判讀〕：
  - (a) repo 的 1.47 是 zstd，不是 ANS 或指數 Huffman。指數編碼（bytes2hi）也有 1.46（S1），最可能對應 DFloat11 式或 ANS 的 GPU 解碼。
  - (b) 解壓佔 SM，GPU 忙時會跟 prefill 搶（repo 第二階段已見 GPU 忙時 H2D 變慢 46–50%）。
  - (c) H100 的 592／480 GB/s 來自單一作者預印本，不是 KV 資料。
  - (d) MI300X 上只有 hipCOMP（無 ANS，early access）與 hipANS（只在 MI100／MI210 測過），見 L0。
  - **這是 §4 裡唯一值得在 H200 上量的一點**：量 BF16 KV 用 nvCOMP ANS 或 DFloat11 式 kernel 的 r 與 D，閒／忙各一次。

### 4.2 會讓縮圖還原的某一步變便宜：FP8 tensor core（H200、MI300X 都有）
- 回來那一輪「新問題在縮圖上 prefill、挑 block」（arith §6、§9）的注意力：強度＝q·G。q＝128–512、G＝4–7 時是 512–3,584 FLOP／byte，遠高於 H200 實測的平衡點 189 FLOP／byte（arith §8）→ **算力型**〔算術〕。
- 用 repo 的 788 TFLOPS 算這一步的純注意力時間〔算術〕：
  - Qwen-14B 256K：q＝128 時 0.042 s、q＝512 時 0.167 s。
  - 照 FA3 的 FP8／BF16 實測比（約 1.6）→ 0.026、0.105 s。
- **條件**：
  - 縮圖本身要能當 FP8 運算元。INT4 或低秩係數還是要先反量化成 FP8／BF16，反量化在 GPU 上是 HBM 型的小成本〔判讀〕。
  - 品質：FA3 用區塊量化＋incoherent processing 才把誤差壓下來（F10）。repo 的 E1 是 FP8 KV 的品質量測，不等於 FP8 注意力運算。
  - MI300X 要用 FNUZ（F16）。
- **跟 decode 無關**：decode 注意力是頻寬型（arith §8：2.6–4 FLOP／byte），FP8 只靠「位元組減半」幫忙，tensor core 算力閒著。

### 4.3 讓縮圖更小，但條件很硬：2:4 稀疏（H200、MI300X 都有）
- 位元組約 0.56 倍（50% 值＋1/16 metadata）、注意力算力最多 2 倍〔文件＋算術〕。
- **條件**：
  - 每 4 個沿內積維的連續元素恰好 2 個是 0，而且只能是第一運算元。
  - 對 QKᵀ，內積維是 head 維度 d，所以是「每個 token 的 128 維裡每 4 維留 2 維」，不是「留一半 token」。
  - 要把 K、Vᵀ 當第一運算元，得轉置兩個 GEMM（HieraSparse Trans-Both）。
- **品質**：HieraSparse 的 magnitude pruning，**壓 K 在 prefill 掉 11.1 分 LongBench，壓 V 掉 1.8 分**〔全文〕。
- **對 Tiara 的意思**〔判讀〕：
  - 縮圖的 V 可以考慮 2:4（1.8 分的代價、0.56 倍位元組）。
  - 縮圖的 K 是拿來挑 block 的，壓 K 的品質代價大。
  - 和既有的 INT4／低秩疊起來時，2:4 的值還要再量化，HieraSparse 沒試。
  - H200、MI300X 上都沒有現成 kernel 或量測。
- 本次沒查 2:4 與量化疊加的論文（查詢：`2:4 sparsity quantization KV cache combined`，未執行）。

### 4.4 只省位元組、不省注意力算力：微縮放 FP4（Blackwell NVFP4、CDNA4 MXFP4）
- NVIDIA 自己的 NVFP4 KV 實作，注意力前先反量化成 FP8（F12）→ 對還原而言等同「更小的低位元格式」。repo 的 INT4（3.2 倍，arith 的 INT4_RATIO）已經涵蓋這個效果〔判讀〕。
- 兩種都不在我們的平台上。**不改變任何既有決定。**

### 4.5 無關（列出理由，以免之後再查）
- **Intel AMX、DSA、IAA、QAT**：我們的兩台主機都是 EPYC；R2 換成 AMX 也翻不了（§3）。
- **EPYC 9684X 3D V-Cache**：每顆 CCD 96 MB、私有，KV 讀一次就走（F5）。
- **Zen 5 的 512-bit**：打平點加倍到 16–28 token，仍 <128（§3）。
- **Blackwell DE**：不在我們的平台；格式只有 LZ4／Snappy／Deflate（F13）。
- **TMA、tcgen05／TMEM**：GPU 內部搬運，不改變 PCIe 還原成本（F10、F12）。
- **CXL 記憶體**：實測頻寬只有理論的 20–47%，比 PCIe 慢（F7）。
- **copy engine 數量**：H200 H2D 已接近 PCIe Gen5 x16 的實用上限（53.55 GiB/s）；MI300X 的瓶頸是軟體（F15、F17）。

### 4.6 新機制的種子（只列，不評估）
- **「短問題就在 CPU 上算」的混合規則**〔判讀〕：q ≤ 4 時 CPU 注意力被頻寬卡住且比 PCIe 快（§3.3）。若有工作負載是極短追問、且回答很短，可以「CPU 先算第一個答案 token，KV 背景搬上 GPU」。這是 NEO 式的想法，agent C 負責查新。
- **GPU 端無損解壓疊在縮圖還原的「精修」傳輸上**：精修只搬 ρ＝1/4–1/8 的 INT4 塊，搬的量已經很小（arith §2：Llama 128K 0.012–0.023 s）。無損再省 1/1.1–1/1.2（X4）只省毫秒級。**比較值得的是「全部載入」那條基線**〔算術＋判讀〕。

---

## §5 查不到／未查證清單

| 項目 | 查詢字串 | 狀態 |
|:--|:--|:--|
| H200／H100 copy engine 數量 | `H100 deviceQuery "copy engine(s)"`；`"H100" "copy engines" number` | 本次搜尋沒找到；要在 H200 上跑 `deviceQuery` |
| Genoa 每顆 CCD 的官方／實測讀頻寬 | `chipsandcheese Genoa per-CCD bandwidth`；AMD 58015 PDF（下載被導向 HTML） | 只找到 Zen 5 的數字〔二手〕 |
| Linux 核心編號是否按 CCD 連續（16 核＝2 顆 CCD） | 沒查 | 〔未查證〕；H200 上 `lscpu -e` 或 `/sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list` 可確認 |
| AMX 實際運作時脈 | Intel 手冊 §20.2.1 只說「通常較低」 | 〔未查證〕 |
| IAA 單顆 Deflate 解壓 GB/s | `"IAA" deflate decompression "GB/s" QPL benchmark single device` | 本次搜尋沒找到 |
| LIA 的平台細節（型號、核數） | 只讀到摘要 | 〔摘要〕 |
| Hopper 上 nvCOMP 各演算法的官方吞吐 | nvCOMP 文件、產品頁 | 只有 Blackwell DE 的 600 GB/s；Hopper 只有單一預印本 |
| MI300X 的 SDMA 是 14 還是 16 個 | ConCCL vs DMA-Latte | 兩個來源不一致 |
| 3090 伺服器 CPU 型號 | repo grep | repo 沒記 |
| ROCm 上用 SMFMAC 做稀疏注意力 | `MI300 sparse attention SMFMAC KV` | 本次搜尋沒找到 |
| Zen 5c／行動版 Zen 5 的 AVX-512 寬度 | 沒深查 | 〔未查證〕（與我們無關） |
| GPUDirect Storage 在 H200 節點（`/dev/md0` RAID）可不可用 | 沒查 | 〔未查證〕；MI300X 的 hipFile 見 L0／D7 |

---

## 參考

### 廠商文件
- [V1] Intel, *Intel® 64 and IA-32 Architectures Optimization Reference Manual Vol. 1*, 248966-051, Oct 2026, §2.1.1, §20.2.1, §20.3 (Table 20-2), §20.5.2. <https://cdrdv2.intel.com/v1/dl/getContent/671488>
- [V2] Intel, *Architecture Instruction Set Extensions and Future Features Programming Reference*, 319433-062, Jun 2026, Table 1-2 & revision history. <https://cdrdv2-public.intel.com/922690/319433-062-architecture-instruction-set-extensions-programming-reference.pdf>
- [V3] AMD, EPYC 9554 / 9124 / 9684X product pages. <https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9554.html>（9124、9684x 同路徑）
- [V4] AMD, "AMD Launches 5th Gen AMD EPYC CPUs…", press release, 2024-10-10. <https://ir.amd.com/news-events/press-releases/detail/1219/amd-launches-5th-gen-amd-epyc-cpus-maintaining-leadership>
- [V5] AMD, *Introducing AMD CDNA™ 3 Architecture* white paper, pp. 8, 24. <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf>
- [V6] AMD, *Introducing AMD CDNA™ 4 Architecture* white paper, pp. 6–8, 18. <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf>
- [V7] AMD ROCm, *Data types and precision support*. <https://rocm.docs.amd.com/en/latest/reference/precision-support.html>
- [V8] AMD ROCm, *hipSPARSELt features for the Instinct MI300 series*. <https://rocm.docs.amd.com/projects/hipSPARSELt/en/latest/conceptual/mi300-features.html>
- [V9] NVIDIA, H200 product page & spec table. <https://www.nvidia.com/en-us/data-center/h200/>
- [V10] NVIDIA, *NVIDIA Hopper Architecture In-Depth* (blog). <https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/>
- [V11] NVIDIA, cuSPARSELt documentation. <https://docs.nvidia.com/cuda/cusparselt/index.html>
- [V12] NVIDIA, nvCOMP *Decompression Engine* FAQ. <https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html>
- [V13] NVIDIA, *Speeding Up Data Decompression with nvCOMP and the NVIDIA Blackwell Decompression Engine* (blog). <https://developer.nvidia.com/blog/speeding-up-data-decompression-with-nvcomp-and-the-nvidia-blackwell-decompression-engine/>
- [V14] NVIDIA, *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference*, 2025-06-24. <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- [V15] NVIDIA, *Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache*, 2025-12-08. <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/>
- [V16] NVIDIA, nvCOMP algorithm selection. <https://docs.nvidia.com/cuda/nvcomp/algorithm_selection.html>

### 量測與論文
- [M1] uops.info, VDPBF16PS (ZMM) / VFMADD231PS (ZMM). <https://uops.info/html-instr/VDPBF16PS_ZMM_ZMM_ZMM.html>
- [M2] Fsas Technologies (Fujitsu), *Performance Report PRIMERGY RX1440 M2*, v1.9, 2026-10-27, STREAM pp. 18–20. <https://sp.ts.fujitsu.com/dmsp/Publications/public/wp-performance-report-primergy-rx1440-m2-ww-en.pdf>；*Memory performance of EPYC 9004/9005…* v1.3. <https://sp.ts.fujitsu.com/dmsp/Publications/public/wp-genoa-memory-performance-ww-en.pdf>
- [M3] Chips and Cheese, *AMD's EPYC 9355P: Inside a 32 Core Zen 5 Server Chip*. <https://chipsandcheese.com/p/amds-epyc-9355p-inside-a-32-core>（〔二手〕，WebFetch 轉述）
- [M4] H. Chen et al., *KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models*, SOSP'25. <https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf>
- [M5] H. Kim et al., *LIA: Cost-efficient LLM Inference Acceleration with Intel AMX and CXL*, ISCA'25. <https://deepmind.google/research/publications/81986/>（〔摘要〕）
- [M6] X. Jiang et al., *NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference*, arXiv 2411.01142. <https://arxiv.org/pdf/2411.01142>
- [M7] J. Shah et al., *FlashAttention-3*, arXiv 2407.08608v2. <https://arxiv.org/html/2407.08608v2>
- [M8] DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv 2412.19437v2, §3.3. <https://arxiv.org/html/2412.19437v2>
- [M9] *HieraSparse: Hierarchical Semi-Structured Sparse KV Attention*, arXiv 2604.16864v2. <https://arxiv.org/html/2604.16864>
- [M10] *Microbenchmarking NVIDIA's Blackwell Architecture: An in-depth Architectural Analysis*, arXiv 2512.02189v3. <https://arxiv.org/html/2512.02189v3>
- [M11] Y. Shavidze, *Compressed-Resident Genomics…*, arXiv 2606.18900v1, §6.4.（單一作者、非同儕審查）<https://arxiv.org/pdf/2606.18900>
- [M12] T. Zhang et al., *70% Size, 100% Accuracy: Lossless LLM Compression… (DFloat11)*, arXiv 2504.11651v3. <https://arxiv.org/html/2504.11651v3>
- [M13] R. Kuper et al., *A Quantitative Analysis and Guidelines of Data Streaming Accelerator in Modern Intel Xeon Scalable Processors*, ASPLOS'24, arXiv 2305.02480v5. <https://arxiv.org/html/2305.02480v5>
- [M14] Y. Sun et al., *Demystifying CXL Memory with Genuine CXL-Ready Systems and Devices*, MICRO'23, arXiv 2303.15375. <https://arxiv.org/pdf/2303.15375>
- [M15] S. Pati et al., *DMA-Latte: Expanding the Reach of DMA Offloads to Latency-bound ML Communication*, arXiv 2511.06605. <https://arxiv.org/pdf/2511.06605>

### 本專案內部
- `docs/stage3_design_20261024/mechanisms/arith.md` §2、§6、§7、§8、§9；`code/s3_mech_arith.py` `section_cpu_attention()`（峰值 16×3.1e9×64）
- `docs/stage3_design_20261024/mechanisms/05_rejected.md:9`（R2）
- `results/RUNLOG_H200.md:63、153、181`；`results/hw_h200.json`（cpuset、co-tenant）
- `docs/H200_SLURM.md` §3 表；`docs/stage3_20261019/environment_snapshot.md` §2–§3；`docs/ENV_3090.md:165`
- `docs/stage3_20261019/experiment_log.md` S1 節；`docs/stage3_20261019/results_analysis.md` §4.5
- `docs/stage3_20261019/lit/L0_intel_suite_and_amd.md` §3.2、E-L1；`L2_compression_offload_prefetch.md:290、377`；`L7c_recheck_amd_restore_path.md` §2.1
- `docs/stage3_20261019/experiment_matrix.md:158`
