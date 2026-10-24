# 05　想過但否決的（每一條都有理由：已被做掉、算術不划算、或我們自己量過）

> **日期**：2026-10-24。列出來是為了不重複想，也讓老師看到範圍有掃過。
> 回到核心頁：[README](README.md)。

| # | 點子 | 為什麼否決 | 證據 |
|:--|:--|:--|:--|
| R1 | P/D 分離時，回來的請求不去 prefill 節點，直接在 decode 節點做「只算新問題」的 prefill，省掉 KV 搬兩次 | **已經被做掉**：PPD（ICML 2026 poster）依情況把第 2 輪以後的請求派到 decode 節點，第 2 輪以後的 TTFT 少 68%；NVIDIA Dynamo 的 conditional disaggregation 也是「prefill 短就在 decode 引擎本地做」 | [07](07_lit_quickcheck.md) 的搜尋結果（摘要級） |
| R2 | 在 CPU 上直接對 CPU 層的 KV 算新問題的注意力，只把結果（每個 query 一個向量）傳回 GPU，不搬 KV | **算術不划算**：每個 GPU job 只有 16 個 EPYC 9554 核、沒有 AMX；算一個問題 token 的注意力要 0.033 s（Qwen-7B 256K）或 0.081 s（14B 256K），而整份 KV 過 PCIe 只要 0.26／0.90 s → 問題超過 7–11 個 token 就輸 | 〔算術〕arith §7（CPU 峰值 3.2 TFLOPS 是假設） |
| R3 | 多個 session 同時回來時，一起決定誰重算、誰載入 | **我們量過**：I1 的 GPU 實測／模擬平均 TTFT 比 0.97–1.05，聯合決定每一格都比各自最佳差 | `docs/stage3_20261019/results_analysis.md` I1 |
| R4 | 寫入時依位置決定層級或精度 | **我們量過**：第一、二階段的負面結果（MI300X 36 組 0 組贏 5%；H200 S5 對 S4+ 在 ±0.5% 內；前／中／後 25% 對隨機都在 ±0.33） | 02_observations O1、O2 |
| R5 | 用 GH200／Grace 的 NVLink-C2C（CPU 和 GPU 之間 900 GB/s）讓 decode 直接讀 CPU 上的 KV | **不是我們的平台**：H200 節點是 x86（2× EPYC 9554）＋PCIe Gen5 | `H200_SLURM.md` |
| R6 | 相鄰層逐 token 共用 KV（只存一半的層） | **我們量過**：相鄰層 K 的 token 向量 cosine ≈ 0（子空間重疊 0.63–0.76，但逐 token 不像） | K1、K1b（02_observations O6） |
| R7 | 依重要性丟 token 或合併 token | **我們量過**：同位元組下全面輸給量化，多輪更糟 | X1、X2、X4d（04_directions F 節） |
| R8 | Tucker 分解本身當新意 | 多出來的維度沒幫助；JoLT、2609.28029 已做 | K1b、L7a |
| R9 | 用影片解碼器（NVDEC）解 KV | 已被 KVCodec 做掉 | L8b |

---

## 和舊點子庫的對照（避免把舊點子當新的）

| 這次的編號 | 舊的出處 | 這次多了什麼 |
|:--|:--|:--|
| M02 逐層挑塊流水 | — | 新 |
| M03 有保證的精修 | — | 新（Quest 的上界只用來排序） |
| M04 打字時挑塊 | 04_directions G2（打字時讀 base） | 從「讀粗底」變成「用部分問題挑塊」 |
| M06 借鄰卡 PCIe | D8-05（多卡同時卸載會互搶，當時判多半被殺） | 方向相反：不是怕互搶，而是借閒著的路；要先量拓撲 |
| M08 搭 decode 的便車 | D8-04（κ 隨同機負載變）、04_directions G7 | 從「量 κ 會變」變成「把 decode 的空檔當預算」 |
| M11 查表式 codec | 04_directions G4 | 補上 DL（學 codebook）與選擇準則 |
| M12 先近似後補精 | D8-11（閒時重算當預取） | 加上「近似重算」這個動作與精度債 |
| M13 賽跑 | D8-03（遠端 KV 的尾延遲保險） | 加上 I3g 的實測證據（P95 105 s） |
| M14 兩段式佇列 | 04_directions G3、方向 E | 品質等級固定，避開產品承諾問題 |
| M15 草稿先行 | D8-10、04_directions G9 | 沒有，只補了算術 |
