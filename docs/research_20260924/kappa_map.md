# κ 地圖：把各篇論文的設備放到同一條 κ 軸上

> **產出日期**：2026-09-24　**範圍**：18 篇論文的評測平台、規格查證、κ／P* 算術、DeepSeek 推理系統的 KV 設計。
> **三種數字分開標**：「論文」＝論文原文（附 PDF 頁碼）；「規格」＝官方或可靠來源（附 URL，見 §8）；「我們的算術」＝本文件用公式算出來的，**不是量測**。
> 平台 A（RTX 3090）與平台 B（MI300X）的實測常數取自專案 `results/`（3090）與 `scratchpad/remote/results/`（MI300X，2026-09-15～09-21），標「實測」。
> **頁碼**＝本地文字檔的 `=== [page N] ===` 標記（PDF 頁）。USENIX 論文的 PDF 頁比印刷頁碼多 1～2 頁。
> 算術腳本與下載的規格原檔：`scratchpad/research/kappa_map_work/`（`kappa_calc.py`、`make_tables.py` 只做算術，不產生資料；`specs/` 為擷取的 datasheet／README 原檔與文字檔）。

---

## 0. 先講結論

1. **只看硬體，κ 的方向跟「邊緣重算、伺服器載回」的直覺相反。** 參考模型 Llama-3.1-8B、MFU 0.5、PCIe 80% 的算術：算力弱的卡 κ 最大——A10 κ_cpu≈50、T4≈48、RTX 3090≈88；算力強的卡 κ 最小——H100 SXM≈12.7、MI300X≈9.6、B200≈5.6。**邊緣卡上重算最貴**，連 SATA 碟的 κ_ssd 都 ≥1（A10 1.08、T4 2.07）。伺服器卡上 κ_ssd 則由碟決定：H100 從 SATA 的 0.14 到 Gen5 NVMe 的 3.6，B200 要 ≥9.2 GB/s 的碟才能在位置 0 贏過重算（§4.1）。
2. **可以用「硬體＋軟體路徑」解釋的：18 篇中有 11 篇的結論與其硬體區間一致**（Strata、Cake、AdaptCache、CachedAttention、HCache、Mooncake、CacheBlend、FlexGen、Bottlenecks，外加方向一致的 EvicPress 與 LMCache）。**有 3 篇與硬體 κ 不一致**：
   * **MTDS**（A10 邊緣）說「長請求改用重算」，方向與物理相反，要用**容量**（VRAM 只給 5 GB 的 KV buffer）與**軟體**來解釋。
   * **Tutti**：硬體 κ_ssd≈7，實測卻是「SSD 比重算慢」，要求端到端軟體路徑再慢約 13 倍以上。
   * **BiDAW**：單請求 κ_ssd≈0.3，要用 **GPU 負載的機會成本**才解釋得通。
   另 4 篇是 decode 期的稀疏讀取或「activation→KV」這類不同的動作（KVDrive、LeoAM、OrbitFlow、KVPR），**無法直接比較**（§4.2）。
3. **軟體路徑是獨立的因子，而且大小與硬體差距同級**（§4.3）：
   * 同一台 H100，LMCache 與 vLLM 原生 CPU offload 差 4.5 倍（LMCache p13）。
   * Bottlenecks 實測 κ_crit＝2，用它自己量到的 15 GB/s 算應為 13（p8）。
   * 平台 B 同一條 PCIe Gen5、同一版 vLLM，CPU 階有效頻寬跨模型從 2.27 到 38.28 GB/s，差 17.1 倍，由每筆描述符 12.9–16.3 µs 的固定成本決定（EXPERIMENTS 實驗 2）。
   * 平台 A 的 SSD 階比原始 I/O 慢 1.3 倍（SATA）、5.0 倍（NVMe，Llama BF16）、18 倍（NVMe，Qwen-AWQ）。
4. **實測 κ 比規格估算低 1.7–13 倍**（§4.5）。
   * 3090：κ_cpu 規格 88.1，實測 6.81（p=0）。
   * MI300X：κ_cpu 規格 9.6，實測 1.12（p=0）。
   * 兩平台實測 κ_cpu 相差 **5.8–6.2 倍**（ctx=16K；p=0 為 6.1 倍）。規格估算相差 9.2 倍，論文 `main.tex` 寫的是 32 倍。
5. **位置效應是真的，但斜率比公式陡。** 實測的 D（注意力讓每 token 重算成本翻倍的位置）：3090 為 19.1K、MI300X 為 16.1K，公式是 30.6K。所以公式會把 P* 放錯位置 1.6–1.9 倍。
6. **Bottlenecks 的 κ_crit 就是本表的 κ_cpu(p=0)**（F_pf＝2N）。它的「99% 延遲在傳輸」是 K/T ≫ κ_crit 的結果，**不等於重算比較好**：實測 κ≈1–2 表示重算同一批 token 仍然比較慢。
7. **DeepSeek**（§5）：
   * MLA 讓 2N/B_kv 比 Llama-8B 大 8.6 倍，同硬體下 κ 也大 8.6 倍。單顆 SATA 碟（0.55 GB/s）就接近「載回＝重算」的門檻（H800、BF16 算力時 BW*＝0.47 GB/s）。
   * 生產端 24 小時內 56.3% 的輸入 token 命中 on-disk KV cache（608B 中的 342B）。
   * V3.2 的 DSA 解碼 kernel 讀 FP8 KV（每層每 token 656 B；FlashMLA）；V4.1 另有 FP4（288 B）。V3/R1 生產系統的 KV 儲存精度，公開文件**沒有說明**。
   * 快取命中的定價是未命中的 2–25%（隨時期不同）。
   * DSA 與壓縮注意力把「重算成本隨位置線性上升」的斜率壓平，讓 P* 這個概念的適用範圍縮小。
8. **對使用者的想法**：「用 κ 把設備列出來再選對應的 SOTA」當**一階地圖**站得住。但 κ 必須從「硬體常數」擴成 **(硬體 × 軟體路徑 × 模型架構 × 位置／ctx × 負載)** 的函數，否則會把 MTDS、Tutti、BiDAW 這三類結論判錯（§6）。

---

## 1. 定義、公式、單位

**κ（每 token）**＝重算一個 token 的 KV ÷ 把它從某一階搬回 GPU。

```
t_rc(p)  = [2N + 4·L·(H_q·d_h)·p] / (F × MFU)          # 位置 p 的一個 token 重算（我們的算術）
t_cpu    = B_kv / BW_host_eff                           # BW_host_eff：論文有實測就用實測，否則 80% × 單向理論值
t_ssd    = B_kv / BW_ssd                                # 論文所述的 SSD 讀頻寬
κ_cpu    = t_rc(0) / t_cpu ;   κ_ssd = t_rc(0) / t_ssd
D        = 2N / (4·L·H_q·d_h)                           # 注意力讓每 token 重算成本翻倍的位置
P*_ssd   = 0                         若 κ_ssd ≥ 1（SSD 永遠較快）
         = D · (1/κ_ssd − 1)          否則
P*_ssd(X)= D · (X/κ_ssd − 1)，X＝SSD 路徑的軟體開銷倍數（X = 1, 2, 4）
整段前綴（長度 C）一次重算 vs 一次載回的交叉點 C* = 2·P*    # 用來對照 LMCache 圖 15 這類「整段」比較
```

* 單位：GB＝10⁹ B、GiB＝2³⁰ B；頻寬一律單向。
* **TP＝G 時**：F＝G×單卡；主機鏈路＝G×單卡（假設各卡獨立 x16，**未查證**）；SSD 用論文給的總頻寬。
* 「參考 Llama-3.1-8B」一律以**單卡**、該平台的單卡鏈路與同一顆 SSD 計算（SSD 頻寬封頂於該卡的 PCIe 單向有效值）。
* **N 的口徑**：dense 用總參數（Llama-3.1-8B 8.03B，含 embedding 與 lm_head，依題述）。MoE 用 active（Qwen3-30B-A3B 3.35B、Qwen3-235B-A22B 22.19B、DeepSeek-V3 37B）。
* 注意力 FLOPs 用 H_q·d_h，Qwen3 這類 H_q·d_h≠d_model 的模型用前者。DeepSeek-V3 prefill（MHA 模式）每層每位置為 2·128·192＋2·128·128＝81,920 FLOPs。

**與其他論文 κ 的對應**：
* Bottlenecks 的 κ_crit＝(F_pf/B_kv)·(BW/C_eff)（p4 式 6），F_pf＝2N 時**等於本表的 κ_cpu(p=0)**。差別在於它的 C_eff 取 ≈2 PFLOP/s（H100；p5 寫「GPU's peak capacity is C_eff ≈2000 TFLOP/s」，等於含 sparsity 的 BF16 或 dense FP8 峰值），所以分析值比 dense BF16 × MFU 0.5 的版本小約 4 倍。
* Cake 的「equivalent throughput」（p4 圖 3，KV 大小 ÷ 計算時間）等於 BW_io/κ。
* CachedAttention 的「generation speed of the KV cache is about 13.9 GB/s」（p4）是同一個概念。

### 1.1 公式與實測的吻合程度（我們的算術）

**平台 A：RTX 3090、Llama-3.1-8B BF16**（題述實測：每 2,048 token chunk＝513 ms＋26.9 ms×P/1000）

| 量 | 實測 | 公式（MFU 0.5，規格 71.2 TFLOPS） | 差 |
|---|---|---|---|
| t_rc(0) | 513/2048＝**250.5 µs/token** | 16.06 GFLOP ÷ 35.6 TFLOP/s＝451.1 µs | 公式**高估 1.80 倍** |
| 位置斜率 | 26.9 ms/(2048×1000)＝**13.13 ns/token/位置** | 524,288 FLOP ÷ 35.6 TFLOP/s＝14.73 ns | 公式高估 1.12 倍 |
| D | 250.5 µs / 13.13 ns＝**19,088 token** | 30,632 token | 公式把 P* 放遠 1.6 倍 |
| 等效 MFU（線性部分） | 2N 口徑 **0.90**（含 chunk 內注意力 0.93）；扣掉 embedding＋lm_head（N'＝6.98B）為 0.78–0.81 | 0.5（假設） | — |
| 等效 MFU（注意力） | 524,288/(13.13 ns×71.2 T)＝**0.56** | 0.5 | — |

* 3090 的線性部分 MFU 高於 0.8。可能原因是 2,048 token 的 GEMM 形狀很好，加上 GeForce 實際 boost 時脈常高於規格的 1,695 MHz（**未查證**）。
* **所以 MFU 0.5 在 3090 上會把 κ 高估約 1.8 倍。** 實測 D 比公式短，也就是注意力相對更貴。

**平台 B：MI300X、Llama-3.1-8B BF16**（實測常數：`cost_model_b-llama8b.json`，由 M3 cold prefill 重新校準）

| 量 | 實測 | 公式（MFU 0.5，規格 1,307.4 TFLOPS） | 差 |
|---|---|---|---|
| t_rc(0) | 0.54698 ms/16 token＝**34.19 µs** | 24.57 µs | 公式**低估 1.39 倍** |
| 位置斜率 | 3.40×10⁻⁵ ms/block/token＝**2.125 ns** | 0.80 ns | 公式低估 2.65 倍 |
| D | **16,087 token** | 30,632 | 1.9 倍 |
| 等效 MFU | 線性 0.36（相對規格）／0.71（相對實測 GEMM 658 TFLOP/s）；注意力 0.19 | 0.5 | — |

* 同一平台的 M2 chunk 擬合（RUNLOG_MI300X §2）是 C0＝101.5 ms/2,048 token＝49.6 µs，a＝9.25 µs/位置/chunk＝4.52 ns/token/位置，等效 MFU 更低（0.25／0.09）。
* **兩種量法差 1.45 倍**，P* 也因此不同（見 §4.5）。
* MI300X 的實測 GEMM 峰值只有規格的 50%（658／1,307.4；ADVISOR_REPLY §1.1），所以「MFU 0.5×規格」恰好等於實測 GEMM 峰值。

**用論文自己的數字驗證**（我們的算術）：
* **Cake** 表 3（p6，2×A100、LongAlpaca-13B、16K、100% util）：由「對 I/O-only 的加速÷對 compute-only 的加速」可反推整段 κ_whole：7 Gbps 0.14、25 Gbps 0.47、32 Gbps 0.61、56 Gbps 1.02、100 Gbps 1.71。公式（MFU 0.5，乘上 16K 平均位置係數 1.26）得 0.11、0.40、0.51、0.90、1.60，**低估 12–20%**，等效 MFU≈0.42–0.45。
* **CachedAttention**（p4，4×A100、LLaMA-65B、2K）：實測重算 360 ms、載回 192 ms，κ＝1.9。公式用其實測 26 GB/s 得 2.08。實測重算相當於 MFU≈0.59。
* **HCache**（p11）：token 重算的恢復速度 1K→16K 下降 28%。公式預測 22.7%（D＝25.7K）；若用 3090 實測的 D/公式 D 比例修正則為 31.7%。實測落在兩者之間。

---

## 2. 規格表（每個數字附 URL，URL 編號見 §8）

### 2.1 GPU

| GPU | dense BF16/FP16 tensor TFLOPS（FP32 累加） | 記憶體／頻寬 | 主機鏈路 | 來源 |
|---|---|---|---|---|
| RTX 3090 | **71.2**（FP16 累加 142.3；sparse ×2） | 24 GB GDDR6X／936 GB/s | PCIe Gen4 x16 | [G1] 表 3（p46–48）；[G2] 表 9（p44–45，列 71） |
| RTX 4090 | **165.2**（FP16 累加 330.3） | 24 GB GDDR6X／1,008 GB/s | PCIe Gen4 | [G3] 附錄 A 表 2（p29–30）；[G1] |
| RTX 5090 | **209.5**（FP16 累加 419） | 32 GB GDDR7／1,792 GB/s | PCIe Gen5 | [G1] 表 3 |
| A10 | **125**（250 sparse） | 24 GB GDDR6／600 GB/s | PCIe Gen4：64 GB/s（雙向） | [G4] |
| A40 | **149.7**（299.4） | 48 GB GDDR6／696 GB/s | PCIe Gen4：64 GB/s | [G5]；[G2] 表 3 |
| RTX A5000 | **111.1**＝222.2÷2（datasheet 只列含 sparsity 的「Tensor performance 222.2」，精度未寫；÷2 是我們的算術） | 24 GB GDDR6／768 GB/s | PCIe 4.0 x16 | [G6] |
| RTX A6000 | **154.8**（309.6） | 48 GB GDDR6／768 GB/s | PCIe Gen4（GA10x） | [G2] 表 3（p15）、p9 |
| T4 | **65**（FP16/FP32 mixed；無 BF16） | 16 GB GDDR6／300 GB/s | PCIe Gen3 x16：32 GB/s（雙向） | [G7] |
| L20 | **119.5**（**非官方**；HCache p9 表 2 列 120T） | 48 GB GDDR6／864 GB/s（非官方） | PCIe Gen4（非官方） | [G8]（第三方彙整）；NVIDIA 未公開規格表，官方來源**未查證** |
| A100 40GB PCIe／80GB PCIe／40GB SXM／80GB SXM | **312**（624） | 1,555／1,935／1,555／2,039 GB/s | PCIe Gen4：64 GB/s；SXM NVLink 600 GB/s | [G9] |
| A800 80GB | **312**（624） | 80 GB／1,935 GB/s（PCIe 版） | PCIe Gen4 64 GB/s；NVLink 400 GB/s；「only difference … NVLink」 | [G10]；[G11]（A800 NVLink 80GB，BF16 312）；SXM 版頻寬**未查證**（推定同 A100 80GB SXM） |
| H100 SXM | **989.5**＝1,979÷2 | 80 GB HBM3／3.35 TB/s | PCIe Gen5：128 GB/s（雙向）；NVLink 900 | [G12]；[G13]「Shown with sparsity. Specifications 1/2 lower without sparsity」 |
| H100 PCIe | **756.5**＝1,513÷2 | 80 GB HBM2e／2 TB/s | PCIe Gen5 x16 | [G13]；[G14] |
| H800 | 論文所列 990T（HCache p9 表 2），官方**未查證** | — | 論文列 64 GB/s | — |
| H200 | **989.5**＝1,979÷2 | 141 GB HBM3e／4.8 TB/s | PCIe Gen5：128 GB/s | [G15] |
| H20 | **148**（dense 或 sparse 未載明；**非官方**） | 96 GB HBM3／4.0 TB/s；NVLink 900 GB/s | PCIe Gen5（非官方） | [G16]（媒體轉述；NVIDIA 未公開規格表） |
| GH200 | H100 級算力（精確值**未查證**，本文以 989.5 計） | HBM3 最高 96 GB、最高 3,000 GB/s；LPDDR5X 最高 512 GB、546 GB/s | **NVLink-C2C 900 GB/s 總計，450 GB/s 單向** | [G17]；[G18]（「900 GB/s coherent interface」） |
| B200 | **2,250**＝HGX B200 8 卡 36 PFLOPS（sparse）÷8÷2（「Dense is ½ sparse」） | 每卡約 180 GB、8 TB/s（DGX 1,440 GB、64 TB/s ÷8） | 單卡 host link **未查證**（Bottlenecks 以 PCIe 5 計） | [G19]；[G20] |
| MI300X | **1,307.4**（2,614.9 sparse） | 192 GB HBM3／5.3 TB/s | PCIe Gen5 x16：128 GB/s；7×IF 128 GB/s | [G21]。**平台 B 實測**：BF16 GEMM 658 TFLOP/s、HBM 3,901 GB/s、H2D 57.6／D2H 48.6 GB/s（2026-09-19 重測；舊 smoke test 為 50.8／48.6） |
| Quadro RTX 5000 | 論文所列 89.2 TFLOPS FP16（KVPR p13），官方**未查證** | 16 GB | 論文：PCIe 4.0 x8，16 GB/s | — |

**GeForce 的半速陷阱**：GeForce 卡 FP16 累加的 tensor 峰值是 FP32 累加的 2 倍（3090 142.3 vs 71.2；4090 330.3 vs 165.2）。框架的 BF16／FP16 GEMM 用 FP32 累加，所以要用後者。**HCache 表 2 的「4090 330T」用的是 FP16 累加值**（p9），拿來算 κ 會把 4090 的算力高估 2 倍。

**PCIe 單向理論值**：Gen3 x16＝16 GB/s（T4 datasheet「32 GB/sec」為雙向 [G7]）；Gen4 x16＝32 GB/s（GA102 whitepaper「up to 64 GB/sec of peak bandwidth」[G2] p9）；Gen5 x16＝64 GB/s（Strata p9「up to 64 GB/s of peak bandwidth (unidirectional)」；H100「PCIe Gen5: 128GB/s」[G12]）。預設有效值取 80%：12.8／25.6／51.2 GB/s。

### 2.2 SSD 與儲存

| 裝置 | 介面 | 循序讀 | 循序寫 | 耐久度 | 來源 | 用在 |
|---|---|---|---|---|---|---|
| Samsung PM9A3（U.2，3.84 TB） | PCIe Gen4 x4 | **6,900 MB/s**（SKU 頁）；product brief 寫「Up to 6,800」（provisional） | 4,100 MB/s（SKU）；brief「Up to 4,000」 | **1.0 DWPD，5 年** | [S1][S2] | HCache 4×4TB（論文 p11：單顆 6.9 GB/s） |
| Samsung 870 QVO | SATA 6 Gb/s | **560 MB/s** | 530 MB/s（Intelligent TurboWrite 內）；**用完 TurboWrite 後 80 MB/s（1 TB）／160 MB/s（2/4/8 TB）** | **TBW 360／720／1,440／2,880 TB**（1/2/4/8 TB），3 年保固 | [S3] | 平台 A `/ssd*`：實測讀 0.49 GB/s；持續寫 181 MiB/s（16 GiB），1 GiB 短測 492 MiB/s 落在 SLC cache |
| Crucial P3 | PCIe Gen3 NVMe M.2 | **3,500 MB/s** | 1,900（500 GB）／3,000 MB/s（1–4 TB） | **TBW 110／220／440／800 TB**（0.5/1/2/4 TB），5 年 | [S4][S5] | 平台 A `/`：實測讀 1.65 GB/s（寫 2,512 MiB/s，專案 `code/m4_ssd_sweep.py` 註記） |
| Intel／Solidigm D7-P5510 | PCIe 4.0 x4 U.2 | up to 7,000 MB/s | up to 4,194 MB/s | 1 DWPD | [S6]（第三方評測列出的規格表） | Strata H20-storage（論文 p9：up to 7 GB/s） |
| Solidigm D7-PS1010 | PCIe 5.0 x4 | **up to 14,500 MB/s** | up to 10,500 MB/s | 1 DWPD（standard endurance） | [S7][S8] | Tutti 4×7.68 TB（論文 p3：2 顆峰值讀 29、寫 12 GB/s；p10：單顆寫 ≤10 GB/s） |
| Samsung 980 PRO（1 TB） | PCIe 4.0 NVMe | 7,000 MB/s | 5,000 MB/s | TBW **未查證**（頁面未列） | [S9] | Cake 的「56 Gbps」情境（模擬） |
| 平台 B 本地碟（Broadcom MegaRAID `GBT3916-MR-32PD`，容器 overlay，**不是 NVMe**，暫態） | — | **4,978 MiB/s**（8 GiB，O_DIRECT）／5,728 MiB/s（64 GiB）；隨機 128K QD8 6,406 MiB/s；4K QD1 166 MB/s | 1,886 MiB/s（RUNLOG_MI300X） | — | 專案實測（ADVISOR_REPLY §1.1、RUNLOG_MI300X） | 平台 B 的 SSD 階（`/var/tmp`） |
| 平台 B NFS（10 GbE，NFSv4.1） | — | 363 MB/s（隨機 128K 118 MiB/s） | 682 MB/s | — | 同上 | 持久儲存；**不適合當 KV 階** |

**論文沒揭露型號、只給數字或完全沒給的碟**：
* MTDS「2 TB SSD」（p10，無頻寬）。
* BiDAW「4 SATA SSDs RAID-5，1.5 GB/s」（p11）。
* CachedAttention「10TB SSDs」「less than 5 GB/s」（p8、p4）。
* KVDrive「NVMe U.2 SSDs」（p15，無頻寬）。
* CacheBlend「1TB NVME SSD，measured 4.8 GB/s」（p10）。
* AdaptCache「400 GB SSD，1 GB/s」（p2）。
* EvicPress「800GB SSD」（p8，無頻寬）。
* LeoAM「800 GB Intel SSD，measured ~7GB/s」（p9）。
* FlexGen「Cloud default SSD (NVMe)，read ~2GB/s，write ~1GB/s」（p7）。
* Cake 完全沒有碟，I/O 延遲是依頻寬「calculating the appropriate delay time」算出來的（p5）。

### 2.3 模型的 KV bytes/token（BF16，由 config.json 算：2·L·H_kv·d_h·2 B；我們的算術）

| 模型 | L | d_model | H_q／H_kv | d_h | **KV B/token（BF16）** | N（B，config 算） | 用在 | config |
|---|---|---|---|---|---|---|---|---|
| Yi-34B | 60 | 7168 | 56／8 | 128 | **245,760**（240 KiB） | 34.39 | CacheBlend | https://huggingface.co/01-ai/Yi-34B/blob/main/config.json |
| Llama-2-13B | 40 | 5120 | 40／40 | 128 | **819,200**（800 KiB） | 13.02 | CachedAttention, HCache, KVPR | https://huggingface.co/NousResearch/Llama-2-13b-hf/blob/main/config.json |
| Llama-2-70B | 80 | 8192 | 64／8 | 128 | **327,680**（320 KiB） | 68.98 | Mooncake(dummy), CachedAttention | https://huggingface.co/NousResearch/Llama-2-70b-hf/blob/main/config.json |
| Llama-2-7B | 32 | 4096 | 32／32 | 128 | **524,288**（512 KiB） | 6.74 | MTDS, HCache, KVPR | https://huggingface.co/NousResearch/Llama-2-7b-hf/blob/main/config.json |
| Llama-3-8B | 32 | 4096 | 32／8 | 128 | **131,072**（128 KiB） | 8.03 | MTDS(LLaMa-3 8B), Tutti, OrbitFlow | https://huggingface.co/NousResearch/Meta-Llama-3-8B/blob/main/config.json |
| Llama-3.1-70B | 80 | 8192 | 64／8 | 128 | **327,680**（320 KiB） | 70.55 | Strata, Cake(FP8 權重), LMCache, Bottlenecks, OrbitFlow(LLaMA3-70B), CacheBlend(Llama-70B) | https://huggingface.co/NousResearch/Meta-Llama-3.1-70B/blob/main/config.json |
| Llama-3.1-8B（官方 repo gated，用公開鏡像的 config） | 32 | 4096 | 32／8 | 128 | **131,072**（128 KiB） | 8.03 | Strata, AdaptCache, EvicPress, Tutti(Llama3-8B), LMCache, Cake, 參考模型 | https://huggingface.co/NousResearch/Meta-Llama-3.1-8B/blob/main/config.json |
| Yarn-Llama-2-13B-128K | 40 | 5120 | 40／40 | 128 | **819,200**（800 KiB） | 13.02 | LeoAM | https://huggingface.co/NousResearch/Yarn-Llama-2-13b-128k/blob/main/config.json |
| Qwen-14B（v1） | 40 | 5120 | 40／40 | 128 | **819,200**（800 KiB） | 14.17 | BiDAW | https://huggingface.co/Qwen/Qwen-14B/blob/main/config.json |
| Qwen-7B（v1） | 32 | 4096 | 32／32 | 128 | **524,288**（512 KiB） | 7.72 | BiDAW | https://huggingface.co/Qwen/Qwen-7B/blob/main/config.json |
| Qwen2.5-14B-Instruct-1M | 48 | 5120 | 40／8 | 128 | **196,608**（192 KiB） | 14.77 | Strata, EvicPress(Qwen2.5-14B) | https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M/blob/main/config.json |
| Qwen2.5-7B | 28 | 3584 | 28／4 | 128 | **57,344**（56 KiB） | 7.62 | 平台 A（AWQ） | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json |
| Qwen3-14B | 40 | 5120 | 40／8 | 128 | **163,840**（160 KiB） | 14.77 | MTDS, KVDrive | https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json |
| Qwen3-235B-A22B（MoE；N 為 active） | 94 | 4096 | 64／4 | 128 | **192,512**（188 KiB） | 22.19（total 235.1） | Bottlenecks | https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json |
| Qwen3-30B-A3B（MoE；N 為 active） | 48 | 2048 | 32／4 | 128 | **98,304**（96 KiB） | 3.35（total 30.5） | EvicPress, 平台 B | https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json |
| Qwen3-32B | 64 | 5120 | 64／8 | 128 | **262,144**（256 KiB） | 32.76 | LMCache(SGLang), Tutti(例子) | https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json |
| Qwen3-8B | 36 | 4096 | 32／8 | 128 | **147,456**（144 KiB） | 8.19 | KVDrive | https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json |
| GLM-4-9B-Chat-1M | 40 | 4096 | 32／4 | 128 | **81,920**（80 KiB） | 9.48 | Tutti | https://huggingface.co/THUDM/glm-4-9b-chat-1m/blob/main/config.json |
| LongAlpaca-13B | 40 | 5120 | 40／40 | 128 | **819,200**（800 KiB） | 13.02 | Cake（論文表 1 列 800 kB） | https://huggingface.co/Yukang/LongAlpaca-13B/blob/main/config.json |
| LongAlpaca-7B | 32 | 4096 | 32／32 | 128 | **524,288**（512 KiB） | 6.74 | Cake（論文表 1 列 512 kB） | https://huggingface.co/Yukang/LongAlpaca-7B/blob/main/config.json |
| OPT-13B | 40 | 5120 | 40／40 | 128 | **819,200**（800 KiB） | 12.84 | BiDAW, KVPR, FlexGen | https://huggingface.co/facebook/opt-13b/blob/main/config.json |
| OPT-30B | 48 | 7168 | 56／56 | 128 | **1,376,256**（1344 KiB） | 29.96 | BiDAW, HCache, KVPR, FlexGen | https://huggingface.co/facebook/opt-30b/blob/main/config.json |
| OPT-6.7B | 32 | 4096 | 32／32 | 128 | **524,288**（512 KiB） | 6.65 | BiDAW, KVPR, LeoAM, FlexGen | https://huggingface.co/facebook/opt-6.7b/blob/main/config.json |
| Llama-3-8B-1048K | 32 | 4096 | 32／8 | 128 | **131,072**（128 KiB） | 8.03 | KVDrive | https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k/blob/main/config.json |
| LLaMA-65B | 80 | 8192 | 64／64 | 128 | **2,621,440**（2560 KiB） | 65.28 | CachedAttention | https://huggingface.co/huggyllama/llama-65b/blob/main/config.json |
| LongChat-7B-v1.5-32K | 32 | 4096 | 32／32 | 128 | **524,288**（512 KiB） | 6.74 | EvicPress, LeoAM | https://huggingface.co/lmsys/longchat-7b-v1.5-32k/blob/main/config.json |
| Phi-4-mini-instruct | 32 | 3072 | 24／8 | 128 | **131,072**（128 KiB） | 3.84 | KVDrive | https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/config.json |
| Mistral-7B-v0.3 | 32 | 4096 | 32／8 | 128 | **131,072**（128 KiB） | 7.25 | EvicPress, CacheBlend, CachedAttention | https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/main/config.json |
| GPT-2 1.5B（XL） | 48 | 1600 | 25／25 | 64 | **307,200**（300 KiB） | 1.55 | MTDS | https://huggingface.co/openai-community/gpt2-xl/blob/main/config.json |
| Falcon-40B | 60 | 8192 | 128／8 | 64 | **122,880**（120 KiB） | 41.83 | CachedAttention | https://huggingface.co/tiiuae/falcon-40b/blob/main/config.json |
| DeepSeek-V3（MLA） | 61 | 7168 | 128 heads；kv_lora_rank 512＋qk_rope 64 | — | **70,272**（BF16，576×2 B×61）；FP8（V3.2 格式 656 B/層）＝40,016 | active 37（total 671，DeepSeek 公布值） | Strata(H20), Bottlenecks(分析), §5 | https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json |
| DeepSeek-V3.2-Exp | 61 | 7168 | 同上＋indexer 64 heads×128 | — | MLA 同上；另有 indexer K cache（每層每 token 128 維，FP8＋scale） | active 37 | §5.5 | https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/config.json |
| DeepSeek-V2（MLA） | 60 | 5120 | 同 V3 | — | **69,120** | active 21（ISCA'25 p4） | §5.2 | https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/config.json |

* 交叉核對：Cake 表 1 列 LongAlpaca-7B 512 kB、-13B 800 kB、Llama-3.1-8B 128 kB、70B 320 kB（p5）；Bottlenecks 表 3 列 Llama-3.1-70B 328 KB、Qwen3-235B 192 KB、DeepSeek-V3 70 KB（p7）；EvicPress 列 Qwen3-30B-A3B 0.0915 GB/1K token、Llama-3.1-8B 0.12 GB/1K token（p10）；ISCA'25 表 1 列 DeepSeek-V3 70.272 KB（p4）——全部與 config 算出的值一致（KB＝10³ 或 KiB 的差別除外）。
* GLM-4-9B-Chat-1M 的 `multi_query_group_num`＝4 → KV＝81,920 B/token。

---

## 3. 逐篇抽出評測平台（論文原文，附頁碼）

| 論文 | GPU（型號×張） | 主機鏈路 | DRAM | SSD（型號／數量／頻寬；實測・規格・模擬） | 網路 | 模型 | 上下文 | 對「重算 vs 載回」「SSD 值不值得」的結論（原文，頁碼） |
|---|---|---|---|---|---|---|---|---|
| **strata2026**（OSDI'26） | 8×H200 NVLink 節點（8B/14B 單卡、70B 4 卡 TP）；8×H20（「H20-storage」）；GH200（1×H100＋Grace）（p9） | PCIe 5.0 x16「up to 64 GB/s (unidirectional)」；page=32 時僅達理論值約 22%（p4）；Strata GPU-assisted I/O 48 GB/s（p6）；GH200 上 Strata-IO 把持續頻寬從 40 提到 150 GB/s（p13） | 1.6 TB（H200 節點），配置 1 TB pinned（GH200 400 GB）（p10）；GH200 LPDDR5X 464 GB、384 GB/s（p9） | H20-storage：Intel P5510「up to 7 GB/s」（規格值，p9）；H200 機碟「<1 GiB/s」（p13） | — | Llama-3.1-8B、Qwen2.5-14B-Instruct-1M、Llama-3.1-70B（p9）；DeepSeek-V3（碟子實驗，p13） | LooGLE 平均 21,613、NarrativeQA 54,797（濾掉 >128K）、ReviewMT 17,708、ShareGPT 681（表 1，p9） | 「recomputation becomes increasingly costly as context length grows, making it an unattractive alternative」（p5）；「Disk storage is not used in most benchmarks except in §5.3.5」（p10）；H200 的碟「bandwidth-limited (<1 GiB/s)」，改在 H20-storage 做（p13） |
| **mtds2026**（Complex & Intelligent Systems 2026） | 4×A10 24 GB；Qwen-3 14B 用 2 卡 TP，其餘單卡（p10） | PCIe Gen4（p10）；「unidirectional bandwidth under ×16 lanes does not exceed 32 GB/s」（p3）；多卡共用 switch／root port（p7） | 64 GB（p10） | 2 TB SSD（型號、頻寬**未揭露**，p10） | — | GPT-2 1.5B、LLaMa-2 7B、LLaMa-3 8B、Qwen-3 14B（p10）；VRAM KV buffer 固定 5 GB、batch 20（p10） | ShareGPT；Random 16K–24K（p11）；消融 1K（p12） | 「KV caches must be reloaded from DRAM or SSD before reuse, incurring transfer latency that may exceed recomputation time」（p1–2）；「as request length increases, MTDS more frequently adopts the recomputation strategy」（p12）；載回時間「non-linear slow-then-fast growth」、重算「grows linearly」（p6） |
| **bidaw2026**（FAST'26） | 1×A800 80 GB（p11） | PCIe Gen4，「around 30 GB/s」（p11） | 200 GB（performance layer）（p11） | 4 顆 SATA SSD RAID-5，**1.5 GB/s**（p11）；另模擬 5 GB/s（p12） | — | OPT-6.7B、Qwen-7B、OPT-13B、Qwen-14B、OPT-30B（皆 MHA）（p11） | 多輪對話平均 22.4 輪（p2）；圖 3 歷史 ≤2,048 token（p4） | CPU–GPU「rarely a bottleneck」、SSD「forming an I/O bottleneck」（p11）；「increasing the SSD bandwidth by several GB/s does not eliminate the SSD bandwidth bottleneck」（p12）；GQA 模型「KV should be cached instead」（p10）；只測「can be handled by one 80GB A800 GPU after eliminating redundant computation」的到達率（p5） |
| **cake2025**（ICML'25） | 2×A100 80 GB NVLink＋EPYC 7763＋2.0 TB；1×H100（型號未載明）＋26 vCPU＋200 GB（p5） | — | 2.0 TB／200 GB（p5） | **無實體碟**：「We simulate the chunk I/O loading process by calculating the appropriate delay time」（p5）；情境 7／25／32（「Lambda Lab SSD read」）／56（「Samsung 980 pro SSD read」）／100 Gbps（表 2，p6） | 模擬 GCP egress 7／25 Gbps、IB 100 Gbps | LongAlpaca-7B（512 kB/token）、-13B（800 kB）、Llama-3.1-8B（128 kB）、Llama-3.1-70B（320 kB，FP8 權重）（表 1，p5） | 4K–16K 每 2K 取樣（p5）；圖 3 用 32K | 「Compute cost increases for later tokens, while I/O cost remains constant」（p4）；平衡時最有利（p6）；「a single scenario where Cake underperforms … short … 7 Gbps」（p7） |
| **adaptcache2025**（arXiv 2509.00105v2） | 1×A100（型號未載明）（p2） | — | 100 GB（p2） | 400 GB SSD，「disk reading throughput is 1 GB/s」（p2） | — | Llama-3.1-8B-Instruct（p2） | LongBench 6 資料集、1,100 contexts（p2） | 「most KV cache hits come from SSD, which is slow to load」（p1）；「Compared to naive prefill and offloading, AdaptCache reduces TTFT by 56%」（p2） |
| **evicpress2025**（arXiv） | 1×H100 80 GB（型號未載明）（p8） | — | 配置 80 GB（p8） | 800 GB SSD（頻寬**未揭露**）；遠端碟「assume … unlimited space」（p8）；圖 2 示意用 20 vs 2 GB/s（p2） | — | Llama-3.1-8B、Qwen2.5-14B、LongChat-7B-32k、Mistral-7B-v0.3、Qwen3-30B-A3B（p8） | LongBench 12 資料集，平均 12K–108K（表 2，p7–8；單位未寫） | 比 prefill 與 eviction-only 快 1.22–1.56×（圖 7，p8）；「KV caches on disk are slower to read, which requires EVICPRESS to further compress」（p10）；「when the loading bandwidths for fast and slow storage devices are similar … benefits … decrease」（p12） |
| **cachedattention2024**（ATC'24） | 4×A100 80 GB（p8） | PCIe Gen4；實測「about 26 GB/s of effective data transmission bandwidth」（p4） | 128 GB（p8） | 10 TB SSDs（型號未揭露，p8）；「less than 5 GB/s」（p4） | — | LLaMA-65B、LLaMA-2 13B/70B、Falcon-40B、Mistral-7B 32K（p8） | ShareGPT 多輪；47%／30% 的 session 超過 2K／4K（p4） | 65B：「prefilling 2K tokens … about 360 ms. In contrast, loading the KV cache of the 2K tokens (5 GB) … about 192 ms」（p4）；「It is essential to ensure that the KV cache to be accessed in the immediate future is always placed in the host memory instead of disks」（p4） |
| **hcache2025**（EuroSys'25） | 4×A100-40G SXM4 NVLink；敏感度實驗用 A30／4090／L20／H800 雲主機（p9） | 論文表 2：A100／A30／4090／L20 為 32 GB/s，H800 64 GB/s（p9） | 256 GB DDR4（p9） | **4×Samsung PM9A3 4TB**；「One PM9A3 SSD provides a read bandwidth of 6.9 GB/s, and using 4 disks can saturate the upstream PCIe bandwidth」（p11） | — | Llama2-7B、Llama2-13B（單卡）、OPT-30B（4 卡 TP）（p9） | 擴到 16K；ShareGPT4、L-Eval（p9） | 「TTFT for recomputation is 20.0-26.0× slower than the ideal case, while KV offloading is 6.5-13.0× slower」（p2）；「In extreme hardware configurations … HCache may not offer any benefits」（p6） |
| **kvdrive2026**（arXiv，2026） | L20 48 GB／H20 96 GB／RTX 4090 24 GB，各單卡（p15） | — | 100／200／120 GB DDR5（p15） | 「NVMe U.2 SSDs」（型號、頻寬**未揭露**，p15） | — | Llama-3-8B-1048K、Qwen3-8B、Qwen3-14B、Phi-4-mini（p15） | 60K–122K（p7 圖 5、p18） | 「Due to the limited bandwidth between GPU and disk … severe throughput degradation」（p7）；加 SSD 後「only a 40% reduction compared to the DRAM-only」（p21）。decode 期稀疏讀取，無重算選項 |
| **tutti2026**（arXiv 2605.03375） | 2×H100 80 GB（型號未載明）＋Xeon 6530 64 核（p8） | 圖 2 設定：「50 GB/s DRAM-HBM bandwidth」（p3） | 512 GB（256 GB pinned）（p8） | **4×Solidigm D7-PS1010 7.68 TB**（p8）；圖 2 設定 2 顆「peak bandwidth of 29 GB/s for read and 12 GB/s for write」（p3）；實測擷取：LMCache-GDS 約 11.9 GB/s、Tutti 最高 25.9 GB/s（p10） | — | Llama3-8B（主）；GLM-4-9B-Chat-1M（2 卡）（p8、p11） | 64K（圖 2）；最長 640K（p11） | CPU-centric 路徑「making KV cache reuse even slower than recomputation」（p2）；「restoring KV cache from SSDs is no longer beneficial (vLLM v0.12.0 vs. v0.17.0)」（p3 圖 2）；「Even with GDS, GPU bubble time remains high at above 70%」（p4）；Tutti「nearly the same inference performance as DRAM-backed LMCache」（p2） |
| **lmcache2025**（arXiv，2025） | 8×H100（GMI Cloud，型號未載明）；圖 15 用 B200（p11、p14） | 表 5：LMCache 自 CPU 載入 400 Gbps，vLLM 原生 88 Gbps（p13） | CPU offload 上限 500 GB（p11） | 本機碟未量化；遠端 server 15 Gbps（p12）；圖 15 網路 32／64／128 Gbps（p14） | 見左 | Llama-3.1-8B/70B、Qwen2.5-72B、Qwen2.5-Coder-32B、Qwen3-Coder-480B-FP8（p10–11） | 每 query 10K（8B 為 20K）（p11）；圖 15 到約 400K | 「outperforms naive prefilling only when the input context length exceeds 256K tokens」（32 Gbps，p14）；「loading should be enabled only when the context length surpasses the crossover point」（p15）；torch.save／load「sub-1GB/s」（p4） |
| **leoam2025**（arXiv 2506.20187） | RTX 4090 24 GB＋i7-14700K（p9） | PCIe 4.0（p9） | 120 GB（p9） | 800 GB Intel SSD，「measured read throughput … around 7GB/s」（p9） | — | LongChat-7B-v1.5-32K、Yarn-Llama-2-13B-128K、OPT-6.7B（p9） | decode；例子 2K（p5）；討論到 150K（p4） | decode 一步「computation latency is only 100 ms, the transmission latency reaches 290 ms」（p5） |
| **mooncake2025**（arXiv v4） | 8×A800-SXM4-80GB／節點，NVLink（p15） | — | 分散式 DRAM 池（p5） | SSD 在池中但未量化 | RDMA「up to 800 Gbps」節點間（p15） | dummy LLaMA2-70B（p15） | ArXiv 8,088、L-Eval 19,019、模擬 16K–128K、真實 7,955（表 2，p15）；trace 平均 7,590（p7） | 「forwards … if the estimated additional prefill time is shorter than the transfer time」「prefer to compute … if the best remote prefix match length is no larger than the current local reusable prefix multiplied by a threshold」（p11）；「waiting for KVCache stored on lower-tier storage may violate the TTFT SLO」（p2） |
| **cacheblend2025**（EuroSys'25） | 2×A40（Runpod）（p10） | — | 128 GB（p10） | 1 TB NVMe，「measured throughput is 4.8 GB/s」（p10）；另有「Slower Disk (4Gbps)」情境（p12） | — | Mistral-7B、Yi-34B（8-bit）、Llama-70B（8-bit，2 卡）（p10） | 約 4K（RAG，p3） | Llama-7B 4K：15% 重算每層 3 ms、NVMe 讀一層 16 ms；Llama-70B：7 ms vs 4 ms（p8）；「store KV caches in slower devices … without increasing the inference delay」（p1） |
| **kvpr2025**（arXiv 2411.17089） | A100 40 GB；低階組 Quadro RTX 5000 16 GB（p6、p13） | PCIe 4.0 x16「32 GB/s」；低階組 PCIe 4.0 x8「16 GB/s」（p6、p13） | — | 無 | — | OPT-6.7B/13B/30B；LLaMa2-7B/13B（p6、p13） | prompt 256–1024（p6） | 「PCIe latency exceeds KV cache recomputation latency by over an order of magnitude」（表 1，p2；此處的重算是 decode 期由 activation 算 K/V） |
| **orbitflow2026**（2026） | RTX A5000 24 GB；4×RTX A6000 48 GB（p9） | PCIe 3.0 x16；PCIe 4.0 x16（p9） | 384 GB；256 GB（p9） | 無 | — | LLaMA3-8B、LLaMA3-70B（p9） | 到 128K（p9–10） | 重算對 decode「still limited to MHA models … GQA … activations are no longer lightweight」（p12） |
| **bottlenecks2026**（MLSys'26） | 8×H100 SXM5 80 GB NVLink 4.0（p7） | PCIe 5.0 x16「peak bidirectional bandwidth: 128 GB/s」（p7）；實測持續「15 GB/s (23% of unidirectional 64 GB/s peak)」（p8） | 2 TB DDR4-3200（p7） | 無（CPU 階） | — | Llama-3.1-70B（328 KB/token）、Qwen3-235B-A22B（192 KB）（p7）；分析含 DeepSeek-V3 70 KB（表 3，p7） | 微基準 K 到 65K；ShareGPT／NarrativeQA／FinQA 的 κ_ratio 中位數 100／5,000／10,000（p6） | 「99% of latency spent on transfers」（p1）；「We measure κcrit values of 2 and 1 versus estimates of 14.3 and 7.8」（p8）；vLLM 0.10.1＋LMCache 0.3.5（p7） |
| **flexgen2023**（ICML'23） | T4 16 GB（GCP）（p7） | 圖示 GPU–CPU 12 GB/s（p2） | 208 GB（p7） | 「Cloud default SSD (NVMe)」1.5 TB，讀約 2 GB/s、寫約 1 GB/s（p7） | — | OPT-6.7B～175B（p7） | prompt 512／1024（p7） | 把權重與 KV 壓縮「to fit into CPU memory to avoid slow disk swapping」（p8）；全程不重算 |

**對專案內部文件的更正**：`SOTA_MATRIX_20260919.md` 寫 Cake 是「控 recompute-vs-load … 唯一一篇」，並把 MTDS 全列為「未知」。本地全文顯示：
* **MTDS** 有明確的三種策略（Full Load／Partial Load／不載回全重算），並以 T_s 對 V_hL/B_w 的比較式選擇（p5–7，式 9）。
* **Mooncake** 以「額外 prefill 時間 vs 傳輸時間」決定轉送或重算（p11）。
* **LMCache** 明寫載回「should be adaptive」（p15）。
所以「唯一一篇」不成立。

---

## 4. κ 計算（全部是我們的算術；MFU 0.5，另給 0.3／0.7）

### 4.1 硬體 κ 地圖（參考模型 Llama-3.1-8B BF16，單卡，p=0）

| GPU | dense BF16/FP16 TFLOPS | 主機鏈路（有效＝80%） | t_rc(0) µs/token | κ_cpu | κ_ssd SATA 0.55 | κ_ssd Gen3 3.5 | κ_ssd Gen4 7 | κ_ssd Gen5 14.5 | P* SATA | P* Gen3 | P* Gen4 | 使 κ_ssd=1 的 SSD 頻寬 BW* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RTX3090 | 71.2 | Gen4（25.6 GB/s） | 451.1 | 88.1 | 1.89 | 12.05 | 24.09 | 49.91 | 0 | 0 | 0 | 0.29 GB/s |
| RTX4090 | 165.2 | Gen4（25.6 GB/s） | 194.4 | 38.0 | 0.82 | 5.19 | 10.38 | 21.51 | 6.9K | 0 | 0 | 0.67 GB/s |
| RTX5090 | 209.5 | Gen5（51.2 GB/s） | 153.3 | 59.9 | 0.64 | 4.09 | 8.19 | 16.96 | 17.0K | 0 | 0 | 0.85 GB/s |
| A10 | 125.0 | Gen4（25.6 GB/s） | 257.0 | 50.2 | 1.08 | 6.86 | 13.72 | 28.43 | 0 | 0 | 0 | 0.51 GB/s |
| A40 | 149.7 | Gen4（25.6 GB/s） | 214.6 | 41.9 | 0.9 | 5.73 | 11.46 | 23.74 | 3.4K | 0 | 0 | 0.61 GB/s |
| A5000 | 111.1 | Gen4（25.6 GB/s） | 289.1 | 56.5 | 1.21 | 7.72 | 15.44 | 31.98 | 0 | 0 | 0 | 0.45 GB/s |
| A6000 | 154.8 | Gen4（25.6 GB/s） | 207.5 | 40.5 | 0.87 | 5.54 | 11.08 | 22.95 | 4.5K | 0 | 0 | 0.63 GB/s |
| T4 | 65.0 | Gen3（12.8 GB/s） | 494.2 | 48.3 | 2.07 | 13.2 | 26.39 | 54.67 | 0 | 0 | 0 | 0.27 GB/s |
| L20 | 119.5 | Gen4（25.6 GB/s） | 268.8 | 52.5 | 1.13 | 7.18 | 14.35 | 29.73 | 0 | 0 | 0 | 0.49 GB/s |
| A100 | 312.0 | Gen4（25.6 GB/s） | 102.9 | 20.1 | 0.43 | 2.75 | 5.5 | 11.39 | 40.3K | 0 | 0 | 1.27 GB/s |
| A800 | 312.0 | Gen4（25.6 GB/s） | 102.9 | 20.1 | 0.43 | 2.75 | 5.5 | 11.39 | 40.3K | 0 | 0 | 1.27 GB/s |
| H100PCIe | 756.5 | Gen5（51.2 GB/s） | 42.5 | 16.6 | 0.18 | 1.13 | 2.27 | 4.7 | 141.3K | 0 | 0 | 3.09 GB/s |
| H100SXM | 989.5 | Gen5（51.2 GB/s） | 32.5 | 12.7 | 0.14 | 0.87 | 1.73 | 3.59 | 194.3K | 4.7K | 0 | 4.04 GB/s |
| H200 | 989.5 | Gen5（51.2 GB/s） | 32.5 | 12.7 | 0.14 | 0.87 | 1.73 | 3.59 | 194.3K | 4.7K | 0 | 4.04 GB/s |
| H20 | 148.0 | Gen5（51.2 GB/s） | 217.0 | 84.8 | 0.91 | 5.8 | 11.59 | 24.01 | 3.0K | 0 | 0 | 0.6 GB/s |
| B200 | 2250.0 | Gen5（51.2 GB/s） | 14.3 | 5.6 | 0.06 | 0.38 | 0.76 | 1.58 | 480.7K | 49.7K | 9.5K | 9.18 GB/s |
| MI300X | 1307.4 | Gen5（51.2 GB/s） | 24.6 | 9.6 | 0.1 | 0.66 | 1.31 | 2.72 | 266.5K | 16.1K | 0 | 5.34 GB/s |
| GH200(H100) | 989.5 | NVLink-C2C 450/dir（360 GB/s） | 32.5 | 89.2 | — | — | — | — | — | — | — | — |

**讀法（我們的算術）**
* **κ_cpu 跨 17 張卡從 5.6（B200）到 88（3090），約 16 倍。** GH200 的 NVLink-C2C 讓它回到 89。
* **算力越強，κ 越小**：B200 的主機鏈路只比 A100 快 2 倍，算力卻快 7.2 倍。這正是 Bottlenecks 表 2 的「newer GPUs more prone to PCIe bottlenecks」（p6）。
* **邊緣卡（T4、A10、A5000、L20、3090）連 SATA 碟都有 κ_ssd≥1。** 使 κ_ssd＝1 的碟頻寬 BW* 只要 0.27–0.51 GB/s。也就是說，**在邊緣，硬體不支持「重算比較好」**。
* **資料中心卡上，碟的等級決定正負號**：H100 SXM 的 BW*＝4.0 GB/s（Gen3 NVMe 就會輸），B200 的 BW*＝9.2 GB/s（Gen4 NVMe 也會輸，P*≈9.5K）。使用者說的「server 跟 NVMe 跟 SSD 又差這麼多」，在這張表上是成立的。

### 4.2 主表：各論文平台

欄位：κ_cpu、κ_ssd 是位置 0 的值；P*_ssd 給 X＝1／2／4（SSD 路徑有 1、2、4 倍軟體開銷時的交叉位置；0＝SSD 永遠比重算快）。「—」表示該階不存在，或論文沒給資料。

| 論文 | 平台（GPU×張；主機鏈路；SSD） | 模型（KV B/token） | κ_cpu | κ_ssd | P*_ssd（X=1/2/4） | 參考 Llama-3.1-8B 單卡：κ_cpu / κ_ssd / P*(X=1) | 論文結論（原文＋頁碼） | 與區間一致嗎？ |
|---|---|---|---|---|---|---|---|---|
| strata2026 | H200 ×1, PCIe5, 1.6TB DRAM; 硬碟 <1 GiB/s；主機 51.2 GB/s（假設 80%×64）；SSD 1.07 GB/s（論文 p13「<1 GiB/s」(上限)） | Llama-3.1-8B（131,072） | 12.7 | 0.27 | 84.5K / 199.7K / 430.0K | （同本列） | 「recomputation becomes increasingly costly as context length grows, making it an unattractive alternative」(p5)；H200 上不用碟（<1 GiB/s，p13），碟只在 H20+P5510 子實驗用（p10,p13） | **一致**：CPU 階 κ_cpu 3.5–45 且其上下文 17–55K 再放大 1.6–2.8 倍；H200 的慢碟 κ_ssd≈0.23–0.33（P*≈62–177K，碟比重算慢）→ 他們不用它；H20+P5510 跑 MLA 的 κ_ssd≈12 → 用碟合理 |
| strata2026 | H200 ×1（同上），主機鏈路用 baseline 實測 22%×64；主機 14.1 GB/s（論文 p4 實測 22%）；SSD 1.07 GB/s（論文 p13 (<1 GiB/s)） | Llama-3.1-8B（131,072） | 3.49 | 0.27 | 84.5K / 199.7K / 430.0K | （同本列） | 同上 | 同上 |
| strata2026 | H200 ×1（同上），Strata GPU-assisted I/O 48 GB/s；主機 48.0 GB/s（論文 p6 實測 48 GB/s）；SSD 1.07 GB/s（論文 p13 (<1 GiB/s)） | Llama-3.1-8B（131,072） | 11.9 | 0.27 | 84.5K / 199.7K / 430.0K | （同本列） | 同上 | 同上 |
| strata2026 | H200 ×1；主機 51.2 GB/s（假設 80%）；SSD 1.07 GB/s（論文 p13 (<1 GiB/s)） | Qwen2.5-14B-1M（196,608） | 15.6 | 0.33 | 62.1K / 154.2K / 338.5K | 12.7 / 0.27 / 84.5K | 同上 | 同上 |
| strata2026 | H200 ×4 TP；主機 205 GB/s（假設 4×80%）；SSD 1.07 GB/s（論文 p13 (<1 GiB/s)） | Llama-3.1-70B（327,680） | 44.6 | 0.23 | 176.5K / 406.8K / 867.5K | 12.7 / 0.27 / 84.5K | 同上 | 同上 |
| strata2026 | H20 ×8 + Intel P5510 (7 GB/s)；主機 410 GB/s（假設 8×80%）；SSD 7.00 GB/s（論文 p9「up to 7 GB/s」） | DeepSeek-V3(MLA,BF16 KV)（70,272） | 729 | 12.4 | 0 / 0 / 0 | 84.8 / 11.6 / 0 | 同上 | 同上 |
| strata2026 | GH200（H100 + NVLink-C2C），Strata-IO 150 GB/s；主機 150 GB/s（論文 p13 實測 150 GB/s）；SSD：無 SSD | Llama-3.1-8B（131,072） | 37.1 | — | — | （同本列） | 同上 | 同上 |
| mtds2026 | A10 ×1（4 卡中 1 張），PCIe4，64GB DRAM，2TB SSD（型號未揭露）；主機 25.6 GB/s（假設 80%）；SSD 0.55 GB/s（假設：SATA 級（型號未揭露）） | Llama-3.1-8B（131,072） | 50.2 | 1.08 | 0 / 26.2K / 83.0K | （同本列） | 「transfer latency may exceed recomputation time」(p1–2)；「as request length increases, MTDS more frequently adopts the recomputation strategy」(p12) | **不一致（以硬體 κ）**：A10 是表中 κ 最大的一群（κ_cpu≈50），且重算成本隨位置上升，「越長越該重算」與物理方向相反。可解釋為**容量／軟體區間**：VRAM KV buffer 只給 5 GB（p10）、4 卡共用 PCIe（p7）、Python I/O 執行緒、SSD 型號未揭露；若 SSD 為 SATA 級且 X≥2，P*≈26K（>其 16–24K 輸入），才會出現「重算較好」 |
| mtds2026 | A10 ×1（同上），SSD 假設 NVMe Gen4；主機 25.6 GB/s（假設 80%）；SSD 7.00 GB/s（假設：NVMe Gen4（型號未揭露）） | Llama-3.1-8B（131,072） | 50.2 | 13.7 | 0 / 0 / 0 | （同本列） | 同上 | 同上 |
| mtds2026 | A10 ×2 TP；主機 51.2 GB/s（假設 2×80%）；SSD 0.55 GB/s（假設：SATA 級） | Qwen3-14B（163,840） | 73.8 | 0.79 | 9.4K / 54.9K / 145.8K | 50.2 / 1.08 / 0 | 同上 | 同上 |
| bidaw2026 | A800 80GB ×1, PCIe4 ~30GB/s, 200GB DRAM, 4×SATA RAID-5 1.5GB/s；主機 30.0 GB/s（論文 p11「around 30 GB/s」）；SSD 1.50 GB/s（論文 p4/p11） | OPT-13B（819,200） | 6.03 | 0.30 | 72.7K / 176.7K / 384.7K | 23.6 / 1.18 / 0 | CPU–GPU「around 30 GB/s … rarely a bottleneck」、SSD 1.5 GB/s「forming an I/O bottleneck」(p11)；兩階快取＋排程勝過全重算的 vLLM | **部分一致**：CPU 階一致（κ_cpu≈6）。SSD 階以**單請求**看不一致：OPT-13B κ_ssd≈0.30、P*≈73K ≫ 2K 上下文，空閒 GPU 上重算反而快約 3 倍；但論文只測「GPU 已被消除重算後才撐得住」的到達率（p5）→ 在 GPU 滿載區間，重算有機會成本，載回才划算。**κ 需加入負載項** |
| bidaw2026 | 同上，模擬 5 GB/s SSD；主機 30.0 GB/s（論文 p11）；SSD 5.00 GB/s（論文 p12 模擬） | OPT-13B（819,200） | 6.03 | 1.00 | 0 / 31.1K / 93.5K | 23.6 / 3.93 / 0 | 同上 | 同上 |
| bidaw2026 | 同上；主機 30.0 GB/s（論文 p11）；SSD 1.50 GB/s（論文 p11） | Qwen-14B(v1)（819,200） | 6.65 | 0.33 | 69.4K / 173.4K / 381.4K | 23.6 / 1.18 / 0 | 同上 | 同上 |
| bidaw2026 | 同上；主機 30.0 GB/s（論文 p11）；SSD 1.50 GB/s（論文 p11） | OPT-30B（1,376,256） | 8.37 | 0.42 | 60.5K / 164.5K / 372.5K | 23.6 / 1.18 / 0 | 同上 | 同上 |
| cake2025 | A100 80GB ×2 NVLink（TP2），I/O 為模擬；主機階：不適用；SSD 4.00 GB/s（論文 p6「32 Gbps (Lambda SSD read)」模擬） | LongAlpaca-13B/Llama-2-13B（819,200） | — | 0.41 | 46.2K / 124.2K / 280.2K | — / 3.14 / 0 | 「Compute cost increases for later tokens, while I/O cost remains constant」(p4)；兩者平衡時最有利（p6）；7 Gbps＋短序列時輸給 compute-only（p7） | **一致，且可反推驗證公式**：由表 3（p6）反推 16K 全序列 κ_whole＝0.14（7 Gbps）、0.47（25）、0.61（32）、1.02（56）、1.71（100 Gbps）；我們公式（MFU 0.5）得 0.11、0.40、0.51、0.90、1.60，低估 12–20%（等效 MFU≈0.42–0.45）。Cake 的甜蜜點正是 κ≈1。注意：I/O 是**模擬**的（p5），不含軟體路徑 |
| cake2025 | 同上，I/O 7 Gbps；主機階：不適用；SSD 0.88 GB/s（論文 p6 模擬） | LongAlpaca-13B/Llama-2-13B（819,200） | — | 0.09 | 324.8K / 681.4K / 1.39M | — / 0.69 / 13.9K | 同上 | 同上 |
| cake2025 | 同上，I/O 100 Gbps；主機階：不適用；SSD 12.5 GB/s（論文 p6 模擬） | LongAlpaca-13B/Llama-2-13B（819,200） | — | 1.27 | 0 / 18.1K / 68.1K | — / 9.82 / 0 | 同上 | 同上 |
| cake2025 | A100 ×2，I/O 56 Gbps (980 Pro)；主機階：不適用；SSD 7.00 GB/s（論文 p6 模擬） | Llama-3.1-8B（131,072） | — | 2.75 | 0 / 0 / 13.9K | — / 5.50 / 0 | 同上 | 同上 |
| cake2025 | H100 ×1（型號未載明，以 PCIe 版計）；主機階：不適用；SSD 4.00 GB/s（論文 p6 模擬 32 Gbps） | LongAlpaca-13B/Llama-2-13B（819,200） | — | 0.34 | 62.8K / 157.3K / 346.5K | — / 1.30 / 0 | 同上 | 同上 |
| adaptcache2025 | A100 ×1, 100GB DRAM, 400GB SSD 1 GB/s；主機 25.6 GB/s（假設 80%）；SSD 1.00 GB/s（論文 p2「1 GB/s」） | Llama-3.1-8B（131,072） | 20.1 | 0.79 | 8.4K / 47.4K / 125.4K | （同本列） | 「most KV cache hits come from SSD, which is slow to load」(p1)；比 prefill/無壓縮卸載少 56% TTFT（p2） | **一致**：A100＋1 GB/s 碟 κ_ssd≈0.79、P*≈8.4K，碟階本身幾乎無利 → 需要有損壓縮把命中移回 DRAM，正是其動機 |
| evicpress2025 | H100 80GB ×1（型號未載明，以 SXM 計）, 80GB DRAM, 800GB SSD；主機 51.2 GB/s（假設 80%）；SSD 2.00 GB/s（論文 p2 圖 2「示意」2 GB/s（實際 SSD 頻寬未揭露）） | Llama-3.1-8B（131,072） | 12.7 | 0.50 | 31.2K / 93.1K / 216.7K | （同本列） | 比 prefill 與 eviction-only 少 1.22–1.56× TTFT（p8 圖 7）；「KV caches on disk … slower to read」需更高壓縮（p10） | **方向一致、無法定量**：實際 SSD 頻寬未揭露；以其示意 2 GB/s（p2）算 κ_ssd≈0.5、P*≈31K，碟階要靠壓縮才划算，與其觀察同向 |
| evicpress2025 | 同上；主機 51.2 GB/s（假設 80%）；SSD 2.00 GB/s（同上（示意）） | Qwen3-30B-A3B（98,304） | 7.05 | 0.28 | 22.4K / 53.3K / 115.2K | 12.7 / 0.50 / 31.2K | 同上 | 同上 |
| cachedattention2024 | A100 80GB ×4, PCIe4 實測 26 GB/s, 128GB DRAM, 10TB SSD <5GB/s；主機 26.0 GB/s（論文 p4 實測 26 GB/s）；SSD 5.00 GB/s（論文 p4「<5 GB/s」(上限)） | LLaMA-65B（2,621,440） | 2.08 | 0.40 | 75.0K / 199.8K / 449.4K | 20.4 / 3.93 / 0 | 載入 2K token（5 GB）192 ms < 重算 360 ms，但「non-negligible」；碟 <5 GB/s，必須把即將用到的 KV 預取到 DRAM（p4） | **一致**：公式 κ_cpu＝2.08（用其實測 26 GB/s）對上實測 360/192＝1.9；κ_ssd≤0.40、P*≥75K ≫ 2–4K 上下文 → 碟在關鍵路徑上一定輸重算，所以他們用排程提示預取。**κ 需加入「可預取性」** |
| cachedattention2024 | 同上；主機 25.6 GB/s（假設 80%）；SSD 5.00 GB/s（論文 p4 (上限)） | Mistral-7B-v0.3（131,072） | 18.1 | 3.55 | 0 / 0 / 3.5K | 20.1 / 3.93 / 0 | 同上 | 同上 |
| hcache2025 | A100-40G SXM4 ×1, 4× PM9A3（受 PCIe4 上限）；主機 25.6 GB/s（假設 80%）；SSD 25.6 GB/s（論文 p11 單顆 6.9 GB/s×4，封頂於 PCIe） | LongChat-7B/Llama-2-7B（524,288） | 4.22 | 4.22 | 0 / 0 / 0 | 20.1 / 20.1 / 0 | 重算 TTFT 比理想慢 20–26×、KV offload 慢 6.5–13×（p2）；IO 與算力極端失衡時 HCache 無益（p6） | **一致**：4×PM9A3（封頂於 PCIe4）κ_ssd≈4.2、1 顆≈1.1 → 載回較快；其「28% 降速（1K→16K）」與我們線性模型的 23% 同量級 |
| hcache2025 | 同上，1 顆 PM9A3；主機 25.6 GB/s（假設 80%）；SSD 6.90 GB/s（論文 p11） | LongChat-7B/Llama-2-7B（524,288） | 4.22 | 1.14 | 0 / 19.5K / 64.7K | 20.1 / 5.42 / 0 | 同上 | 同上 |
| hcache2025 | A100 ×4 TP；主機 102 GB/s（假設）；SSD 27.6 GB/s（論文 p11） | OPT-30B（1,376,256） | 7.14 | 1.93 | 0 / 1.7K / 46.9K | 20.1 / 20.1 / 0 | 同上 | 同上 |
| kvdrive2026 | L20 48GB ×1, 100GB DDR5, NVMe U.2（型號未揭露）；主機 25.6 GB/s（假設 80%（L20 PCIe4 非官方））；SSD 7.00 GB/s（假設 NVMe Gen4） | Llama-3.1-8B（131,072） | 52.5 | 14.3 | 0 / 0 / 0 | （同本列） | 「Due to the limited bandwidth between GPU and disk … severe throughput degradation」(p7)；加 SSD 後只降 40% 吞吐（p21） | **無法直接判斷**：decode 期稀疏讀取，沒有「重算」選項；SSD 型號未揭露。以 NVMe Gen4 假設，κ_ssd≈5–17，與「SSD 值得做」同向 |
| kvdrive2026 | H20 96GB ×1, 200GB DDR5；主機 51.2 GB/s（假設 80%）；SSD 7.00 GB/s（假設 NVMe Gen4） | Qwen3-14B（163,840） | 125 | 17.1 | 0 / 0 / 0 | 84.8 / 11.6 / 0 | 同上 | 同上 |
| kvdrive2026 | RTX 4090 ×1, 120GB DDR5；主機 25.6 GB/s（假設 80%）；SSD 7.00 GB/s（假設 NVMe Gen4） | Phi-4-mini（131,072） | 18.2 | 4.97 | 0 / 0 / 0 | 38.0 / 10.4 / 0 | 同上 | 同上 |
| tutti2026 | H100 80GB ×1（型號未載明，以 SXM 計）, DRAM-HBM 50GB/s, 2× Solidigm PS1010 29GB/s；主機 50.0 GB/s（論文 p3「50 GB/s DRAM-HBM」）；SSD 29.0 GB/s（論文 p3 峰值 29 GB/s（2 顆）） | Llama-3.1-8B（131,072） | 12.4 | 7.18 | 0 / 0 / 0 | （同本列） | CPU-centric 的 SSD 路徑「making KV cache reuse even slower than recomputation」(p2)；vLLM 0.17 上「restoring KV cache from SSDs is no longer beneficial」(p3)；GPU-centric 後 SSD≈DRAM（p2） | **與硬體 κ 不一致、與含軟體路徑的 κ 一致**：兩顆 PS1010（29 GB/s）κ_ssd≈7.2；即使用 LMCache-GDS 的實測擷取 11.9 GB/s 仍≈3。要讓 64K 請求「比重算慢」，端到端路徑須再慢約 13× 以上（相對 29 GB/s）。這是**軟體路徑是獨立因子**最強的外部證據 |
| tutti2026 | 同上，SSD 用 LMCache-GDS 實測擷取頻寬；主機 50.0 GB/s（論文 p3）；SSD 11.9 GB/s（論文 p10 實測 11.9 GB/s） | Llama-3.1-8B（131,072） | 12.4 | 2.95 | 0 / 0 / 10.9K | （同本列） | 同上 | 同上 |
| tutti2026 | 同上，SSD 用 Tutti 實測擷取頻寬；主機 50.0 GB/s（論文 p3）；SSD 25.9 GB/s（論文 p10 實測 25.9 GB/s） | Llama-3.1-8B（131,072） | 12.4 | 6.41 | 0 / 0 / 0 | （同本列） | 同上 | 同上 |
| lmcache2025 | H100 ×1（GMI Cloud，型號未載明，以 SXM 計），LMCache CPU 載入 400 Gbps；主機 50.0 GB/s（論文 p13 表 5 實測 400 Gbps）；SSD 1.88 GB/s（論文 p12 遠端 15 Gbps） | Llama-3.1-8B（131,072） | 12.4 | 0.46 | 35.3K / 101.3K / 233.2K | （同本列） | 32 Gbps 時「outperforms naive prefilling only when the input context length exceeds 256K」、64/128 Gbps 全勝（p14）；CPU 載入 LMCache 400 Gbps vs vLLM 原生 88 Gbps（p13 表 5） | **方向一致；定量需 X≈1.6–2.3**：以 Llama-8B（圖 15 模型未載明）算，32 Gbps 的逐位置 P*≈40K、整段交叉≈2P*≈79K，論文觀察 256K。同硬體上軟體路徑讓 κ_cpu 從 12.4 掉到 2.7（4.5×） |
| lmcache2025 | 同上，vLLM 原生 CPU offload 88 Gbps；主機 11.0 GB/s（論文 p13 表 5 實測 88 Gbps）；SSD 1.88 GB/s（論文 p12） | Llama-3.1-8B（131,072） | 2.72 | 0.46 | 35.3K / 101.3K / 233.2K | （同本列） | 同上 | 同上 |
| lmcache2025 | B200 ×1，遠端 32 Gbps（圖 15，模型未載明）；主機 51.2 GB/s（假設 80%）；SSD 4.00 GB/s（論文 p14 圖 15（32 Gbps）） | Llama-3.1-8B（131,072） | 5.58 | 0.44 | 39.7K / 110.0K / 250.6K | （同本列） | 同上 | 同上 |
| lmcache2025 | B200 ×1，遠端 64 Gbps；主機 51.2 GB/s（假設）；SSD 8.00 GB/s（論文 p14 圖 15） | Llama-3.1-8B（131,072） | 5.58 | 0.87 | 4.5K / 39.7K / 110.0K | （同本列） | 同上 | 同上 |
| lmcache2025 | B200 ×1，遠端 128 Gbps；主機 51.2 GB/s（假設）；SSD 16.0 GB/s（論文 p14 圖 15） | Llama-3.1-8B（131,072） | 5.58 | 1.74 | 0 / 4.5K / 39.7K | （同本列） | 同上 | 同上 |
| leoam2025 | RTX 4090 ×1, 120GB, Intel SSD 實測 ~7 GB/s；主機 25.6 GB/s（假設 80%）；SSD 7.00 GB/s（論文 p9 實測 ~7 GB/s） | LongChat-7B/Llama-2-7B（524,288） | 7.97 | 2.18 | 0 / 0 / 21.5K | 38.0 / 10.4 / 0 | decode 一步「computation latency is only 100 ms, the transmission latency reaches 290 ms」(p5) | **無法直接判斷**：decode 期、無重算選項；κ_ssd≈2.2（LongChat-7B, MHA） |
| mooncake2025 | A800 SXM4 ×8/節點，RDMA 800 Gbps（遠端 DRAM 池）；主機 80.0 GB/s（假設 800 Gbps×80%（遠端 DRAM，非本機 PCIe））；SSD：SSD 未量化 | Llama-2-70B(dummy)（327,680） | 27.0 | — | — | 7.85 / — / — | 「forwards … if the estimated additional prefill time is shorter than the transfer time」、遠端前綴不夠長就「prefer to compute」(p11)；低階儲存等待可能違反 TTFT SLO（p2） | **一致**：每節點 RDMA 800 Gbps、Llama-2-70B（GQA）κ≈27 → 以快取為中心；重算只在壅塞／熱點時勝出 → **κ 要用當下有效頻寬，不是規格** |
| cacheblend2025 | A40 ×1, 128GB RAM, 1TB NVMe 實測 4.8 GB/s；主機 25.6 GB/s（假設 80%）；SSD 4.80 GB/s（論文 p10 實測 4.8 GB/s） | Mistral-7B-v0.3（131,072） | 37.8 | 7.09 | 0 / 0 / 0 | 41.9 / 7.86 / 0 | Llama-7B 4K：重算 15% 每層 3 ms < NVMe 讀一層 16 ms；Llama-70B：7 ms > 4 ms（p8）；可把 KV 放更慢的裝置（p1） | **一致**：A40＋4.8 GB/s 對 Mistral-7B κ_ssd≈7；其每層數字反推「全重算」的 κ：Llama-7B（MHA）≈1.25、Llama-70B（GQA）≈12 → 與 κ 隨 GQA/模型規模上升的方向相同 |
| cacheblend2025 | 同上，Slower Disk 4 Gbps；主機 25.6 GB/s（假設）；SSD 0.50 GB/s（論文 p12 圖（4 Gbps）） | Mistral-7B-v0.3（131,072） | 37.8 | 0.74 | 9.8K / 47.2K / 122.0K | 41.9 / 0.82 / 6.8K | 同上 | 同上 |
| cacheblend2025 | A40 ×2（Llama-70B 8-bit，以 BF16 算力計）；主機 51.2 GB/s（假設）；SSD 4.80 GB/s（論文 p10） | Llama-3.1-70B（327,680） | 147 | 13.8 | 0 / 0 / 0 | 41.9 / 7.86 / 0 | 同上 | 同上 |
| kvpr2025 | A100 40GB ×1, PCIe4 x16；主機 25.6 GB/s（論文 p6 32 GB/s ×80%）；SSD：無 SSD | OPT-6.7B（524,288） | 4.16 | — | — | 20.1 / — / — | 「PCIe latency exceeds KV cache recomputation latency by over an order of magnitude」(p2 表 1) | **定義不同，無法直接判斷**：其表 1 的「recomputation」是 decode 期由已存 activation 產生 K/V 的 GPU 計算，不是從 token 重算整段前綴。從 token 重算的 κ_cpu≈4.2（A100, OPT-6.7B）→ 載回仍較快。它屬於第三種動作（activation→KV，與 HCache 同類）；我們的算術：投影只需 4d²L≈2.1 GFLOP/token（≈13.8 µs，MFU 0.5），低於搬整份 KV 的 20.5 µs，方向與其結論相同 |
| kvpr2025 | Quadro RTX 5000 ×1, PCIe4 x8 16 GB/s；主機 12.8 GB/s（論文 p13 16 GB/s×80%）；SSD：無 SSD | OPT-6.7B（524,288） | 7.28 | — | — | 35.2 / — / — | 同上 | 同上 |
| orbitflow2026 | RTX A5000 ×1, PCIe3 x16, 384GB；主機 12.8 GB/s（假設 80%）；SSD：無 SSD | Llama-3.1-8B（131,072） | 28.2 | — | — | （同本列） | 重算對 decode 的效益「still limited to MHA models … GQA … activations are no longer lightweight」(p12) | **無法直接判斷**（decode 期卸載、無 SSD）；κ_cpu≈28（A5000, PCIe3）～142（4×A6000, 70B）→ 不考慮重算合理 |
| orbitflow2026 | RTX A6000 ×4, PCIe4 x16, 256GB；主機 102 GB/s（假設）；SSD：無 SSD | Llama-3.1-70B（327,680） | 142 | — | — | 40.5 / — / — | 同上 | 同上 |
| bottlenecks2026 | H100 SXM5（κ 與 TP 數無關），PCIe5 規格；主機 51.2 GB/s（假設 80%）；SSD：無 SSD | Llama-3.1-70B（327,680） | 44.6 | — | — | 12.7 / — / — | 「99% of latency spent on transfers」(p1)；實測 κ_crit＝2（Llama-70B）、1（Qwen3-235B），估計值 14.3、7.8（p8）；實測 PCIe 只有 15 GB/s（23%） | **一致（且 κ_crit≡本表 κ_cpu(p=0)）**：99% 是 K/T ≫ κ_crit 的結果，**不代表重算比較好**——實測 κ≈1–2 表示重算同一批 K 仍慢 1–2 倍（長位置更多）。我們的規格 κ_cpu＝44.6、用其實測 15 GB/s＝13.1，實測 2 → 軟體路徑再壓 6.5×。注意其 C_eff 取 ≈2 PFLOP/s（含 sparsity 或 FP8 的峰值），分析值因此偏低 |
| bottlenecks2026 | 同上，PCIe 用論文實測 15 GB/s；主機 15.0 GB/s（論文 p8 實測 15 GB/s）；SSD：無 SSD | Llama-3.1-70B（327,680） | 13.1 | — | — | 3.71 / — / — | 同上 | 同上 |
| bottlenecks2026 | 同上；主機 15.0 GB/s（論文 p8 實測 15 GB/s）；SSD：無 SSD | Qwen3-235B-A22B（192,512） | 6.99 | — | — | 3.71 / — / — | 同上 | 同上 |
| flexgen2023 | T4 16GB (GCP), 208GB DRAM, NVMe 讀 2 GB/s；主機 12.0 GB/s（論文 p2 圖「12 GB/s」）；SSD 2.00 GB/s（論文 p7「about 2GB/s」） | OPT-30B（1,376,256） | 16.1 | 2.68 | 0 / 0 / 21.5K | 45.2 / 7.54 / 0 | 把權重與 KV 壓縮「to fit into CPU memory to avoid slow disk swapping」(p8)；不重算 | **一致**：T4 κ_cpu≈16、κ_ssd≈2.7（OPT-30B）→ 載回永遠優於重算，碟只作容量補充 |
| flexgen2023 | 同上；主機 12.0 GB/s（論文 p2）；SSD 2.00 GB/s（論文 p7） | OPT-175B（4,718,592） | 27.4 | 4.56 | 0 / 0 / 0 | 45.2 / 7.54 / 0 | 同上 | 同上 |

**判定的共同規則**：κ 遠離 1（≥3 或 ≤0.3）時，MFU ±2 倍也不會改變結論，判定穩健。κ 在 0.5–2 之間時，結論對 MFU、X、ctx 都敏感，我們只判「方向」。

### 4.3 軟體路徑倍數 X：已知的實測與反推

| 來源 | 平台 | 路徑 | 硬體（原始）頻寬 | 有效頻寬 | X | 性質 |
|---|---|---|---|---|---|---|
| Tutti p3、p10 | H100＋2×PS1010 | LMCache-GDS 擷取（微基準） | 29 GB/s（峰值） | 11.9 GB/s | 2.4 | 論文實測 |
| Tutti p2–3 | 同上 | LMCache-SSD 端到端（64K、75% 命中，「比重算慢」） | 29 GB/s | 須 <2.3 GB/s | **約 13 以上** | 我們由論文結論反推 |
| Tutti p10 | 同上 | Tutti（GPU-centric） | 29 GB/s | 25.9 GB/s | 1.1 | 論文實測 |
| Bottlenecks p8 | H100 SXM，PCIe5 | vLLM 0.10.1＋LMCache 0.3.5 | 64 GB/s | 15 GB/s | 4.3（對峰值） | 論文實測 |
| Bottlenecks p8 | 同上 | 端到端 κ_crit | κ（15 GB/s）＝13.1 | 實測 κ＝2 | **再 6.5** | 我們的算術 |
| LMCache p13 表 5 | H100（推定） | CPU→GPU：LMCache vs vLLM 原生 | — | 50 vs 11 GB/s | **4.5（同硬體）** | 論文實測 |
| Strata p4、p6 | H200，PCIe5 | page=32 的 baseline；Strata I/O | 64 GB/s | 約 14（22%）；48 GB/s | 4.5；1.3 | 論文實測 |
| Strata p13 | GH200，C2C | SGLang-HiCache；Strata-IO | 450 GB/s（單向） | 40；150 GB/s | 11；3 | 論文實測 |
| LMCache p4 | — | torch.save／load | — | 「sub-1GB/s」 | — | 論文 |
| **平台 A（3090）** | PCIe4 | vLLM OffloadingConnector CPU 階（Llama BF16） | 25.6 GB/s（80%） | 3.56 GB/s | **7.2** | 實測（`cost_model.json`，0.588 ms/2 MiB block） |
| 平台 A | SATA 870 QVO | SSD 階（Llama BF16） | 0.49 GB/s（原始讀） | 0.379 GB/s | **1.29**（非 I/O 23%） | 實測（5.536 ms/block） |
| 平台 A | NVMe Crucial P3 | SSD 階（Llama BF16） | 1.65 GB/s | 0.334 GB/s | **4.95**（非 I/O 80%） | 由實測 P*＝10,851 反推 |
| 平台 A | NVMe Crucial P3 | SSD 階（Qwen2.5-7B AWQ） | 1.65 GB/s | 0.090 GB/s | **18.4**（非 I/O 95%） | 實測（10.245 ms/block） |
| **平台 B（MI300X）** | PCIe Gen5 | CPU 階（Llama-8B） | 57.6 GB/s（實測 H2D） | 4.29（TTFT）／4.47 GB/s（connector） | **13.4／12.9** | 實測 |
| 平台 B | 同上 | CPU 階，跨 5 個模型 | 57.6 GB/s | 2.27–38.28 GB/s | 1.5–25（**跨模型 17.1×**） | 實測（ADVISOR_REPLY §1.2） |
| 平台 B | MegaRAID overlay | SSD 階（Llama-8B） | 5.22 GB/s（4,978 MiB/s） | 2.02 GB/s | **2.6**（非 I/O 61%） | 實測 |

* **平台 B 的機制已查明**：ROCm 上 vLLM 停用 Triton 快速路徑，改走 `ops.swap_blocks_batch`（`hipMemcpyBatchAsync`，受 ROCm 7.2.1 bug 限制只能用 `numAttrs=0`）。每筆描述符有 **12.9 µs（32 KiB）／16.3 µs（1.5 MiB）** 的固定成本，與 payload 無關；每筆 4 MiB 時才達 48.5 GB/s（EXPERIMENTS_20260919 實驗 2）。
* 用這個機制回推每 block 的成本：Llama 32 層×64 KiB ≈ 0.45 ms，實測 0.469 ms；Qwen3-MoE 合併成一筆 1.5 MiB ≈ 0.044 ms，實測 0.041 ms。
* **平台 A 是否同機制**：3090 CPU 階換算成每層每筆 18.4 µs（Llama）／10.7 µs（Qwen-AWQ）。量級相同，但**未在 3090 上做微基準，只是假說**。
* 平台 A 的「SSD 階非 I/O 佔比」有兩種口徑：
  * RUNLOG 發現 11（SATA 整段讀取 2.18 s ÷ 約 5.7 s）≈ 35% 是 I/O、65% 是軟體。
  * 上表用每 block 常數得 23%（SATA）／80%（NVMe）／95%（Qwen-AWQ）。
  * 協調者訊息提到的「85%」我在專案檔中**沒找到出處**，未採用。
* **量測陷阱**：平台 B 同一檔案 O_DIRECT 2.7 GB/s，有 page cache 時 10.8 GB/s（膨脹 4 倍；ADVISOR_REPLY §1.1）。論文若沒寫 O_DIRECT，碟的頻寬可能被高估。

### 4.4 MFU 敏感度（論文模型列；κ ∝ 1/MFU）

<details><summary>展開（52 列）</summary>

| 論文 | 平台 | 模型 | κ_cpu (MFU 0.3 / 0.5 / 0.7) | κ_ssd (0.3 / 0.5 / 0.7) | P*_ssd X=1 (0.3 / 0.5 / 0.7) |
|---|---|---|---|---|---|
| strata2026 | H200 ×1, PCIe5, 1.6TB DRAM; 硬碟 <1 GiB/s | Llama-3.1-8B | 21.1 / 12.7 / 9.06 | 0.44 / 0.27 / 0.19 | 38.5K / 84.5K / 130.6K |
| strata2026 | H200 ×1（同上），主機鏈路用 baseline 實測 22%×64 | Llama-3.1-8B | 5.81 / 3.49 / 2.49 | 0.44 / 0.27 / 0.19 | 38.5K / 84.5K / 130.6K |
| strata2026 | H200 ×1（同上），Strata GPU-assisted I/O 48 G | Llama-3.1-8B | 19.8 / 11.9 / 8.49 | 0.44 / 0.27 / 0.19 | 38.5K / 84.5K / 130.6K |
| strata2026 | H200 ×1 | Qwen2.5-14B-1M | 25.9 / 15.6 / 11.1 | 0.54 / 0.33 / 0.23 | 25.2K / 62.1K / 98.9K |
| strata2026 | H200 ×4 TP | Llama-3.1-70B | 74.3 / 44.6 / 31.8 | 0.39 / 0.23 / 0.17 | 84.4K / 176.5K / 268.6K |
| strata2026 | H20 ×8 + Intel P5510 (7 GB/s) | DeepSeek-V3(MLA,BF16 KV) | 1,214 / 729 / 520 | 20.8 / 12.4 / 8.89 | 0 / 0 / 0 |
| strata2026 | GH200（H100 + NVLink-C2C），Strata-IO 150 G | Llama-3.1-8B | 61.9 / 37.1 / 26.5 | — / — / — | — / — / — |
| mtds2026 | A10 ×1（4 卡中 1 張），PCIe4，64GB DRAM，2TB SSD | Llama-3.1-8B | 83.7 / 50.2 / 35.9 | 1.80 / 1.08 / 0.77 | 0 / 0 / 9.1K |
| mtds2026 | A10 ×1（同上），SSD 假設 NVMe Gen4 | Llama-3.1-8B | 83.7 / 50.2 / 35.9 | 22.9 / 13.7 / 9.80 | 0 / 0 / 0 |
| mtds2026 | A10 ×2 TP | Qwen3-14B | 123 / 73.8 / 52.8 | 1.32 / 0.79 / 0.57 | 0 / 9.4K / 27.6K |
| bidaw2026 | A800 80GB ×1, PCIe4 ~30GB/s, 200GB DRAM, | OPT-13B | 10.1 / 6.03 / 4.31 | 0.50 / 0.30 / 0.22 | 31.1K / 72.7K / 114.3K |
| bidaw2026 | 同上，模擬 5 GB/s SSD | OPT-13B | 10.1 / 6.03 / 4.31 | 1.67 / 1.00 / 0.72 | 0 / 0 / 12.3K |
| bidaw2026 | 同上 | Qwen-14B(v1) | 11.1 / 6.65 / 4.75 | 0.55 / 0.33 / 0.24 | 27.8K / 69.4K / 111.0K |
| bidaw2026 | 同上 | OPT-30B | 13.9 / 8.37 / 5.98 | 0.70 / 0.42 / 0.30 | 18.9K / 60.5K / 102.1K |
| cake2025 | A100 80GB ×2 NVLink（TP2），I/O 為模擬 | LongAlpaca-13B/Llama-2-13B | — / — / — | 0.68 / 0.41 / 0.29 | 15.0K / 46.2K / 77.4K |
| cake2025 | 同上，I/O 7 Gbps | LongAlpaca-13B/Llama-2-13B | — / — / — | 0.15 / 0.09 / 0.06 | 182.2K / 324.8K / 467.4K |
| cake2025 | 同上，I/O 100 Gbps | LongAlpaca-13B/Llama-2-13B | — / — / — | 2.12 / 1.27 / 0.91 | 0 / 0 / 3.2K |
| cake2025 | A100 ×2，I/O 56 Gbps (980 Pro) | Llama-3.1-8B | — / — / — | 4.58 / 2.75 / 1.96 | 0 / 0 / 0 |
| cake2025 | H100 ×1（型號未載明，以 PCIe 版計） | LongAlpaca-13B/Llama-2-13B | — / — / — | 0.56 / 0.34 / 0.24 | 24.9K / 62.8K / 100.6K |
| adaptcache2025 | A100 ×1, 100GB DRAM, 400GB SSD 1 GB/s | Llama-3.1-8B | 33.5 / 20.1 / 14.4 | 1.31 / 0.79 / 0.56 | 0 / 8.4K / 24.0K |
| evicpress2025 | H100 80GB ×1（型號未載明，以 SXM 計）, 80GB DRAM,  | Llama-3.1-8B | 21.1 / 12.7 / 9.06 | 0.83 / 0.50 / 0.35 | 6.5K / 31.2K / 55.9K |
| evicpress2025 | 同上 | Qwen3-30B-A3B | 11.8 / 7.05 / 5.04 | 0.46 / 0.28 / 0.20 | 10.0K / 22.4K / 34.8K |
| cachedattention2024 | A100 80GB ×4, PCIe4 實測 26 GB/s, 128GB DR | LLaMA-65B | 3.46 / 2.08 / 1.48 | 0.67 / 0.40 / 0.29 | 25.1K / 75.0K / 124.9K |
| cachedattention2024 | 同上 | Mistral-7B-v0.3 | 30.3 / 18.1 / 13.0 | 5.91 / 3.55 / 2.53 | 0 / 0 / 0 |
| hcache2025 | A100-40G SXM4 ×1, 4× PM9A3（受 PCIe4 上限） | LongChat-7B/Llama-2-7B | 7.03 / 4.22 / 3.01 | 7.03 / 4.22 / 3.01 | 0 / 0 / 0 |
| hcache2025 | 同上，1 顆 PM9A3 | LongChat-7B/Llama-2-7B | 7.03 / 4.22 / 3.01 | 1.90 / 1.14 / 0.81 | 0 / 0 / 5.9K |
| hcache2025 | A100 ×4 TP | OPT-30B | 11.9 / 7.14 / 5.10 | 3.21 / 1.93 / 1.38 | 0 / 0 / 0 |
| kvdrive2026 | L20 48GB ×1, 100GB DDR5, NVMe U.2（型號未揭露） | Llama-3.1-8B | 87.5 / 52.5 / 37.5 | 23.9 / 14.3 / 10.2 | 0 / 0 / 0 |
| kvdrive2026 | H20 96GB ×1, 200GB DDR5 | Qwen3-14B | 208 / 125 / 89.1 | 28.4 / 17.1 / 12.2 | 0 / 0 / 0 |
| kvdrive2026 | RTX 4090 ×1, 120GB DDR5 | Phi-4-mini | 30.3 / 18.2 / 13.0 | 8.28 / 4.97 / 3.55 | 0 / 0 / 0 |
| tutti2026 | H100 80GB ×1（型號未載明，以 SXM 計）, DRAM-HBM 50 | Llama-3.1-8B | 20.6 / 12.4 / 8.84 | 12.0 / 7.18 / 5.13 | 0 / 0 / 0 |
| tutti2026 | 同上，SSD 用 LMCache-GDS 實測擷取頻寬 | Llama-3.1-8B | 20.6 / 12.4 / 8.84 | 4.91 / 2.95 / 2.11 | 0 / 0 / 0 |
| tutti2026 | 同上，SSD 用 Tutti 實測擷取頻寬 | Llama-3.1-8B | 20.6 / 12.4 / 8.84 | 10.7 / 6.41 / 4.58 | 0 / 0 / 0 |
| lmcache2025 | H100 ×1（GMI Cloud，型號未載明，以 SXM 計），LMCache | Llama-3.1-8B | 20.6 / 12.4 / 8.84 | 0.77 / 0.46 / 0.33 | 8.9K / 35.3K / 61.7K |
| lmcache2025 | 同上，vLLM 原生 CPU offload 88 Gbps | Llama-3.1-8B | 4.54 / 2.72 / 1.95 | 0.77 / 0.46 / 0.33 | 8.9K / 35.3K / 61.7K |
| lmcache2025 | B200 ×1，遠端 32 Gbps（圖 15，模型未載明） | Llama-3.1-8B | 9.29 / 5.58 / 3.98 | 0.73 / 0.44 / 0.31 | 11.6K / 39.7K / 67.8K |
| lmcache2025 | B200 ×1，遠端 64 Gbps | Llama-3.1-8B | 9.29 / 5.58 / 3.98 | 1.45 / 0.87 / 0.62 | 0 / 4.5K / 18.6K |
| lmcache2025 | B200 ×1，遠端 128 Gbps | Llama-3.1-8B | 9.29 / 5.58 / 3.98 | 2.90 / 1.74 / 1.24 | 0 / 0 / 0 |
| leoam2025 | RTX 4090 ×1, 120GB, Intel SSD 實測 ~7 GB/s | LongChat-7B/Llama-2-7B | 13.3 / 7.97 / 5.69 | 3.63 / 2.18 / 1.56 | 0 / 0 / 0 |
| mooncake2025 | A800 SXM4 ×8/節點，RDMA 800 Gbps（遠端 DRAM 池） | Llama-2-70B(dummy) | 45.0 / 27.0 / 19.3 | — / — / — | — / — / — |
| cacheblend2025 | A40 ×1, 128GB RAM, 1TB NVMe 實測 4.8 GB/s | Mistral-7B-v0.3 | 63.1 / 37.8 / 27.0 | 11.8 / 7.09 / 5.07 | 0 / 0 / 0 |
| cacheblend2025 | 同上，Slower Disk 4 Gbps | Mistral-7B-v0.3 | 63.1 / 37.8 / 27.0 | 1.23 / 0.74 / 0.53 | 0 / 9.8K / 24.7K |
| cacheblend2025 | A40 ×2（Llama-70B 8-bit，以 BF16 算力計） | Llama-3.1-70B | 245 / 147 / 105 | 23.0 / 13.8 / 9.86 | 0 / 0 / 0 |
| kvpr2025 | A100 40GB ×1, PCIe4 x16 | OPT-6.7B | 6.94 / 4.16 / 2.97 | — / — / — | — / — / — |
| kvpr2025 | Quadro RTX 5000 ×1, PCIe4 x8 16 GB/s | OPT-6.7B | 12.1 / 7.28 / 5.20 | — / — / — | — / — / — |
| orbitflow2026 | RTX A5000 ×1, PCIe3 x16, 384GB | Llama-3.1-8B | 47.1 / 28.2 / 20.2 | — / — / — | — / — / — |
| orbitflow2026 | RTX A6000 ×4, PCIe4 x16, 256GB | Llama-3.1-70B | 237 / 142 / 102 | — / — / — | — / — / — |
| bottlenecks2026 | H100 SXM5（κ 與 TP 數無關），PCIe5 規格 | Llama-3.1-70B | 74.3 / 44.6 / 31.8 | — / — / — | — / — / — |
| bottlenecks2026 | 同上，PCIe 用論文實測 15 GB/s | Llama-3.1-70B | 21.8 / 13.1 / 9.33 | — / — / — | — / — / — |
| bottlenecks2026 | 同上 | Qwen3-235B-A22B | 11.7 / 6.99 / 4.99 | — / — / — | — / — / — |
| flexgen2023 | T4 16GB (GCP), 208GB DRAM, NVMe 讀 2 GB/s | OPT-30B | 26.8 / 16.1 / 11.5 | 4.47 / 2.68 / 1.91 | 0 / 0 / 0 |
| flexgen2023 | 同上 | OPT-175B | 45.6 / 27.4 / 19.6 | 7.61 / 4.56 / 3.26 | 0 / 0 / 0 |

</details>

### 4.5 實測校準：平台 A（3090）、平台 B（MI300X）——實測 vs 規格估算

**Llama-3.1-8B BF16，每 token；「規格」＝MFU 0.5、PCIe 80%（3090）或 51.2 GB/s（MI300X）、原始碟頻寬（X＝1）**

| 量 | 平台 A 實測（3090） | 平台 A 規格估算 | 差（規格÷實測） | 平台 B 實測（MI300X） | 平台 B 規格估算 | 差（規格÷實測） |
|---|---|---|---|---|---|---|
| t_rc(0) µs | 250.5 | 451.1 | 1.80 | 34.19 | 24.57 | 0.72 |
| 位置斜率 ns | 13.12 | 14.73 | 1.12 | 2.125 | 0.80 | 0.38 |
| D（token） | 19,088 | 30,632 | 1.60 | 16,087 | 30,632 | 1.90 |
| CPU 階 µs（有效 GB/s） | 36.77（3.56） | 5.12（25.6） | 0.14（實測慢 7.2×） | 30.53（4.29） | 2.56（51.2）；2.28（實測 H2D 57.6） | 0.084／0.075（實測慢 11.9／13.4×） |
| SSD 階 µs（有效 GB/s） | SATA 346.0（0.379）；NVMe 392.9（0.334） | SATA 267.5（0.49）；NVMe 79.4（1.65） | 0.77；0.20 | 64.92（2.02） | 25.1（5.22，MegaRAID O_DIRECT） | 0.39 |
| **κ_cpu(0)** | **6.81** | 88.1 | **12.9×** | **1.12** | 9.6（規格鏈路）／10.8（實測 H2D） | **8.6×／9.6×** |
| **κ_ssd(0)** | SATA **0.72**；NVMe **0.64** | SATA 1.69；NVMe 5.68 | 2.3×；8.9× | **0.53** | 0.98 | 1.85× |
| κ at ctx＝16K（CLAIM_EVIDENCE B1） | κ_cpu 8.85（SATA run）／9.53（NVMe run）；κ_ssd 0.94／0.82 | κ_cpu 111.7；κ_ssd 2.14／7.2 | 12.6×；2.3×／8.8× | κ_cpu **1.53**；κ_ssd **0.72** | κ_cpu 12.2；κ_ssd 1.24 | 8.0×；1.7× |
| **P*_ssd** | SATA **7,278**；NVMe **10,851** | 0；0 | — | **14,461**（cost model 常數，我們的算術）；**3,396**（RUNLOG_MI300X M2 chunk 法） | ≈0.7K | — |

* **跨平台**：
  * 實測 κ_cpu 相差 6.81/1.12＝**6.1 倍**（p=0）；ctx 16K 時 8.85–9.53 ÷ 1.53＝**5.8–6.2 倍**。
  * 規格估算相差 88.1/9.6＝9.2 倍。論文 `main.tex` 寫 32 倍（CLAIM_EVIDENCE_MI300X B1）。
  * 拆開來看：重算的跨平台比，規格是 18.4 倍，實測只剩 7.3 倍（250.5/34.19）；CPU 階規格是 2 倍，實測只剩 1.2 倍（36.77/30.53）。**κ 的跨平台差距被實作路徑壓縮了。**
* **平台 A 的 NVMe P* 大於 SATA P***（10,851 > 7,278），等於 NVMe 路徑比 SATA 還貴，與原始頻寬相反。RUNLOG 發現 11 記錄了 NVMe 那次 `lookup_async_delay_seconds_sum` 累計 733 s、根分割區共用且已用 88%，「標為待查，不作為結論依據」。10,851 是否出自同一批量測，我沒有在專案檔中找到明確對應。**在釐清前，這兩個 P* 不宜當作「換碟」的比較。**
* **平台 B 的兩個 P* 相差 4.3 倍**（14,461 vs 3,396），差在重算常數的量法。cost model 由 M3 cold prefill 重新校準，M2 是「已快取 P 後再算 2,048 token」的 chunk 法。論文引用時必須標明量法。
* **平台 B 其他模型**（cost model 常數，p=0；我們的算術）：

| 模型 | t_rc(0) µs | 斜率 ns | D | CPU 階 GB/s | SSD 階 GB/s | κ_cpu(0) | κ_ssd(0) | P* |
|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 34.19 | 2.125 | 16,087 | 4.29 | 2.02 | 1.12 | 0.53 | 14,461 |
| UltraLong-8B-1M | 33.07 | 2.194 | 15,075 | 4.29 | 2.02 | 1.08 | 0.51 | 14,436 |
| Mistral-Nemo-12B | 46.53 | 2.704 | 17,207 | 4.30 | 1.93 | 1.22 | 0.55 | 14,246 |
| Qwen2.5-14B-1M | 57.07 | 4.007 | 14,241 | 4.81 | 2.30 | 1.40 | 0.67 | 7,100 |
| Qwen2.5-7B-1M | 29.48 | 1.679 | 17,556 | 2.44 | 1.29 | 1.26 | 0.66 | 8,891 |
| Qwen3-30B-A3B（MoE） | 26.33 | 3.331 | 7,904 | **38.28** | 3.18 | **10.25** | 0.85 | 1,384 |
| Seed-OSS-36B | 141.44 | 10.686 | 13,235 | 4.65 | 2.18 | 2.51 | 1.18 | 0（SSD 永遠較快） |

→ **同一張 MI300X 上 κ_cpu 跨模型差 9.5 倍（1.08–10.25），主要來自描述符粒度（§4.3）**，大小與跨硬體差距同級。

* **平台 A 的 Qwen2.5-7B AWQ**（NVMe；實測常數 `qwen-awq/cost_model.json`）：t_rc(0) 221.6 µs、斜率 11.10 ns、D 19,965、CPU 階 3.07 GB/s、SSD 階 0.090 GB/s；κ_cpu 11.9、κ_ssd 0.35、P* 37,717。AWQ 是 W4A16，算力模型與 BF16 不同，所以不做規格估算。

### 4.6 CSV（主表的完整數值，含 MFU 0.3／0.7 與 X＝1／2／4）

```csv
paper,platform,which,model,gpu,n_gpu,F_TFLOPS_total,kv_bytes,t_rc0_us,D_tokens,host_GBps,host_src,ssd_GBps,ssd_src,kappa_cpu,kappa_ssd,Pstar_X1,Pstar_X2,Pstar_X4,kappa_cpu_mfu0.3,kappa_cpu_mfu0.7,kappa_ssd_mfu0.3,kappa_ssd_mfu0.7,Pstar_X1_mfu0.3,Pstar_X1_mfu0.7,note,verdict_short
strata2026,"H200 ×1, PCIe5, 1.6TB DRAM; 硬碟 <1 GiB/s",論文模型,Llama-3.1-8B,H200,1,989.5,131072,32.46,30632,51.2,假設 80%×64,1.074,論文 p13「<1 GiB/s」(上限),12.68,0.27,84533,199699,430029,21.13,9.06,0.44,0.19,38467,130599,,一致
strata2026,H200 ×1（同上），主機鏈路用 baseline 實測 22%×64,論文模型,Llama-3.1-8B,H200,1,989.5,131072,32.46,30632,14.08,論文 p4 實測 22%,1.074,論文 p13 (<1 GiB/s),3.49,0.27,84533,199699,430029,5.81,2.49,0.44,0.19,38467,130599,SGLang-HiCache page=32,一致
strata2026,H200 ×1（同上），Strata GPU-assisted I/O 48 GB/s,論文模型,Llama-3.1-8B,H200,1,989.5,131072,32.46,30632,48.0,論文 p6 實測 48 GB/s,1.074,論文 p13 (<1 GiB/s),11.89,0.27,84533,199699,430029,19.81,8.49,0.44,0.19,38467,130599,,一致
strata2026,H200 ×1,論文模型,Qwen2.5-14B-1M,H200,1,989.5,196608,59.71,30049,51.2,假設 80%,1.074,論文 p13 (<1 GiB/s),15.55,0.33,62083,154215,338479,25.91,11.11,0.54,0.23,25230,98935,,一致
strata2026,H200 ×1,參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,H200,1,989.5,131072,32.46,30632,51.2,假設 80%,1.074,論文 p13 (<1 GiB/s),12.68,0.27,84533,199699,430029,21.13,9.06,0.44,0.19,38467,130599,,一致
strata2026,H200 ×4 TP,論文模型,Llama-3.1-70B,H200,4,3958.0,327680,71.3,53825,204.8,假設 4×80%,1.074,論文 p13 (<1 GiB/s),44.56,0.23,176505,406836,867497,74.27,31.83,0.39,0.17,84373,268637,,一致
strata2026,H20 ×8 + Intel P5510 (7 GB/s),論文模型,"DeepSeek-V3(MLA,BF16 KV)",H20,8,1184.0,70272,125.0,14808,409.6,假設 8×80%,7.0,論文 p9「up to 7 GB/s」,728.6,12.45,0,0,0,1214.33,520.43,20.75,8.89,0,0,H20 規格非官方,一致
strata2026,H20 ×8 + Intel P5510 (7 GB/s),參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,H20,1,148.0,131072,217.03,30632,51.2,假設 8×80%,7.0,論文 p9「up to 7 GB/s」,84.78,11.59,0,0,0,141.29,60.55,19.32,8.28,0,0,H20 規格非官方,一致
strata2026,GH200（H100 + NVLink-C2C），Strata-IO 150 GB/s,論文模型,Llama-3.1-8B,GH200(H100),1,989.5,131072,32.46,30632,150.0,論文 p13 實測 150 GB/s,,無 SSD,37.15,,,,,61.91,26.53,,,,,,一致
mtds2026,A10 ×1（4 卡中 1 張），PCIe4，64GB DRAM，2TB SSD（型號未揭露）,論文模型,Llama-3.1-8B,A10,1,125.0,131072,256.96,30632,25.6,假設 80%,0.55,假設：SATA 級（型號未揭露）,50.19,1.08,0,26186,83004,83.65,35.85,1.8,0.77,0,9141,LLaMa-3 8B,不一致（以硬體 κ）
mtds2026,A10 ×1（同上），SSD 假設 NVMe Gen4,論文模型,Llama-3.1-8B,A10,1,125.0,131072,256.96,30632,25.6,假設 80%,7.0,假設：NVMe Gen4（型號未揭露）,50.19,13.72,0,0,0,83.65,35.85,22.87,9.8,0,0,,不一致（以硬體 κ）
mtds2026,A10 ×2 TP,論文模型,Qwen3-14B,A10,2,250.0,163840,236.32,36059,51.2,假設 2×80%,0.55,假設：SATA 級,73.85,0.79,9395,54850,145759,123.08,52.75,1.32,0.57,0,27577,Qwen-3 14B,不一致（以硬體 κ）
mtds2026,A10 ×2 TP,參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,A10,1,125.0,131072,256.96,30632,25.6,假設 2×80%,0.55,假設：SATA 級,50.19,1.08,0,26186,83004,83.65,35.85,1.8,0.77,0,9141,Qwen-3 14B,不一致（以硬體 κ）
bidaw2026,"A800 80GB ×1, PCIe4 ~30GB/s, 200GB DRAM, 4×SATA RAID-5 1.5GB/s",論文模型,OPT-13B,A800,1,312.0,819200,164.62,31347,30.0,論文 p11「around 30 GB/s」,1.5,論文 p4/p11,6.03,0.3,72652,176652,384652,10.05,4.31,0.5,0.22,31052,114252,,部分一致
bidaw2026,"A800 80GB ×1, PCIe4 ~30GB/s, 200GB DRAM, 4×SATA RAID-5 1.5GB/s",參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,A800,1,312.0,131072,102.95,30632,30.0,論文 p11「around 30 GB/s」,1.5,論文 p4/p11,23.56,1.18,0,21368,73368,39.27,16.83,1.96,0.84,0,5768,,部分一致
bidaw2026,同上，模擬 5 GB/s SSD,論文模型,OPT-13B,A800,1,312.0,819200,164.62,31347,30.0,論文 p11,5.0,論文 p12 模擬,6.03,1.0,0,31052,93452,10.05,4.31,1.67,0.72,0,12332,,部分一致
bidaw2026,同上，模擬 5 GB/s SSD,參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,A800,1,312.0,131072,102.95,30632,30.0,論文 p11,5.0,論文 p12 模擬,23.56,3.93,0,0,568,39.27,16.83,6.55,2.81,0,0,,部分一致
bidaw2026,同上,論文模型,Qwen-14B(v1),A800,1,312.0,819200,181.67,34594,30.0,論文 p11,1.5,論文 p11,6.65,0.33,69405,173405,381405,11.09,4.75,0.55,0.24,27805,111005,,部分一致
bidaw2026,同上,論文模型,OPT-30B,A800,1,312.0,1376256,384.1,43538,30.0,論文 p11,1.5,論文 p11,8.37,0.42,60462,164462,372462,13.95,5.98,0.7,0.3,18862,102062,,部分一致
cake2025,A100 80GB ×2 NVLink（TP2），I/O 為模擬,論文模型,LongAlpaca-13B/Llama-2-13B,A100,2,624.0,819200,83.46,31787,,非主機階,4.0,論文 p6「32 Gbps (Lambda SSD read)」模擬,,0.41,46213,124213,280213,,,0.68,0.29,15013,77413,,一致，且可反推驗證公式
cake2025,A100 80GB ×2 NVLink（TP2），I/O 為模擬,參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,A100,1,312.0,131072,102.95,30632,,非主機階,4.0,論文 p6「32 Gbps (Lambda SSD read)」模擬,,3.14,0,0,8368,,,5.24,2.24,0,0,,一致，且可反推驗證公式
cake2025,同上，I/O 7 Gbps,論文模型,LongAlpaca-13B/Llama-2-13B,A100,2,624.0,819200,83.46,31787,,,0.875,論文 p6 模擬,,0.09,324784,681356,1394499,,,0.15,0.06,182156,467413,,一致，且可反推驗證公式
cake2025,同上，I/O 7 Gbps,參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,A100,1,312.0,131072,102.95,30632,,,0.875,論文 p6 模擬,,0.69,13939,58511,147654,,,1.15,0.49,0,31768,,一致，且可反推驗證公式
cake2025,同上，I/O 100 Gbps,論文模型,LongAlpaca-13B/Llama-2-13B,A100,2,624.0,819200,83.46,31787,,,12.5,論文 p6 模擬,,1.27,0,18133,68053,,,2.12,0.91,0,3157,,一致，且可反推驗證公式
cake2025,同上，I/O 100 Gbps,參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,A100,1,312.0,131072,102.95,30632,,,12.5,論文 p6 模擬,,9.82,0,0,0,,,16.36,7.01,0,0,,一致，且可反推驗證公式
cake2025,A100 ×2，I/O 56 Gbps (980 Pro),論文模型,Llama-3.1-8B,A100,2,624.0,131072,51.47,30632,,,7.0,論文 p6 模擬,,2.75,0,0,13939,,,4.58,1.96,0,0,,一致，且可反推驗證公式
cake2025,A100 ×2，I/O 56 Gbps (980 Pro),參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,A100,1,312.0,131072,102.95,30632,,,7.0,論文 p6 模擬,,5.5,0,0,0,,,9.16,3.93,0,0,,一致，且可反推驗證公式
cake2025,H100 ×1（型號未載明，以 PCIe 版計）,論文模型,LongAlpaca-13B/Llama-2-13B,H100PCIe,1,756.5,819200,68.84,31787,,,4.0,論文 p6 模擬 32 Gbps,,0.34,62775,157338,346463,,,0.56,0.24,24950,100600,,一致，且可反推驗證公式
cake2025,H100 ×1（型號未載明，以 PCIe 版計）,參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,H100PCIe,1,756.5,131072,42.46,30632,,,4.0,論文 p6 模擬 32 Gbps,,1.3,0,16649,63930,,,2.16,0.93,0,2465,,一致，且可反推驗證公式
adaptcache2025,"A100 ×1, 100GB DRAM, 400GB SSD 1 GB/s",論文模型,Llama-3.1-8B,A100,1,312.0,131072,102.95,30632,25.6,假設 80%,1.0,論文 p2「1 GB/s」,20.11,0.79,8368,47368,125368,33.51,14.36,1.31,0.56,0,23968,,一致
evicpress2025,"H100 80GB ×1（型號未載明，以 SXM 計）, 80GB DRAM, 800GB SSD",論文模型,Llama-3.1-8B,H100SXM,1,989.5,131072,32.46,30632,51.2,假設 80%,2.0,論文 p2 圖 2「示意」2 GB/s（實際 SSD 頻寬未揭露）,12.68,0.5,31212,93055,216743,21.13,9.06,0.83,0.35,6474,55949,,方向一致、無法定量
evicpress2025,同上,論文模型,Qwen3-30B-A3B,H100SXM,1,989.5,98304,13.54,8519,51.2,假設 80%,2.0,同上（示意）,7.05,0.28,22402,53324,115168,11.76,5.04,0.46,0.2,10034,34771,,方向一致、無法定量
evicpress2025,同上,參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,H100SXM,1,989.5,131072,32.46,30632,51.2,假設 80%,2.0,同上（示意）,12.68,0.5,31212,93055,216743,21.13,9.06,0.83,0.35,6474,55949,,方向一致、無法定量
cachedattention2024,"A100 80GB ×4, PCIe4 實測 26 GB/s, 128GB DRAM, 10TB SSD <5GB/s",論文模型,LLaMA-65B,A100,4,1248.0,2621440,209.23,49804,26.0,論文 p4 實測 26 GB/s,5.0,論文 p4「<5 GB/s」(上限),2.08,0.4,74995,199795,449395,3.46,1.48,0.67,0.29,25075,124915,,一致
cachedattention2024,"A100 80GB ×4, PCIe4 實測 26 GB/s, 128GB DRAM, 10TB SSD <5GB/s",參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,A100,1,312.0,131072,102.95,30632,26.0,論文 p4 實測 26 GB/s,5.0,論文 p4「<5 GB/s」(上限),20.42,3.93,0,0,568,34.04,14.59,6.55,2.81,0,0,,一致
cachedattention2024,同上,論文模型,Mistral-7B-v0.3,A100,1,312.0,131072,92.95,27656,25.6,假設 80%,5.0,論文 p4 (上限),18.15,3.55,0,0,3543,30.26,12.97,5.91,2.53,0,0,,一致
cachedattention2024,同上,參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,A100,1,312.0,131072,102.95,30632,25.6,假設 80%,5.0,論文 p4 (上限),20.11,3.93,0,0,568,33.51,14.36,6.55,2.81,0,0,,一致
hcache2025,"A100-40G SXM4 ×1, 4× PM9A3（受 PCIe4 上限）",論文模型,LongChat-7B/Llama-2-7B,A100,1,312.0,524288,86.41,25711,25.6,假設 80%,25.6,論文 p11 單顆 6.9 GB/s×4，封頂於 PCIe,4.22,4.22,0,0,0,7.03,3.01,7.03,3.01,0,0,Llama2-7B,一致
hcache2025,"A100-40G SXM4 ×1, 4× PM9A3（受 PCIe4 上限）",參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,A100,1,312.0,131072,102.95,30632,25.6,假設 80%,25.6,論文 p11 單顆 6.9 GB/s×4，封頂於 PCIe,20.11,20.11,0,0,0,33.51,14.36,33.51,14.36,0,0,Llama2-7B,一致
hcache2025,同上，1 顆 PM9A3,論文模型,LongChat-7B/Llama-2-7B,A100,1,312.0,524288,86.41,25711,25.6,假設 80%,6.9,論文 p11,4.22,1.14,0,19506,64724,7.03,3.01,1.9,0.81,0,5941,,一致
hcache2025,同上，1 顆 PM9A3,參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,A100,1,312.0,131072,102.95,30632,25.6,假設 80%,6.9,論文 p11,20.11,5.42,0,0,0,33.51,14.36,9.03,3.87,0,0,,一致
hcache2025,A100 ×4 TP,論文模型,OPT-30B,A100,4,1248.0,1376256,96.03,43538,102.4,假設,27.6,論文 p11,7.14,1.93,0,1679,46896,11.91,5.1,3.21,1.38,0,0,,一致
kvdrive2026,"L20 48GB ×1, 100GB DDR5, NVMe U.2（型號未揭露）",論文模型,Llama-3.1-8B,L20,1,119.5,131072,268.79,30632,25.6,假設 80%（L20 PCIe4 非官方）,7.0,假設 NVMe Gen4,52.5,14.35,0,0,0,87.5,37.5,23.92,10.25,0,0,Llama-3-8B-1048K,無法直接判斷
kvdrive2026,"H20 96GB ×1, 200GB DDR5",論文模型,Qwen3-14B,H20,1,148.0,163840,399.19,36059,51.2,假設 80%,7.0,假設 NVMe Gen4,124.75,17.06,0,0,0,207.91,89.1,28.43,12.18,0,0,,無法直接判斷
kvdrive2026,"H20 96GB ×1, 200GB DDR5",參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,H20,1,148.0,131072,217.03,30632,51.2,假設 80%,7.0,假設 NVMe Gen4,84.78,11.59,0,0,0,141.29,60.55,19.32,8.28,0,0,,無法直接判斷
kvdrive2026,"RTX 4090 ×1, 120GB DDR5",論文模型,Phi-4-mini,RTX4090,1,165.2,131072,92.98,19531,25.6,假設 80%,7.0,假設 NVMe Gen4,18.16,4.97,0,0,0,30.27,12.97,8.28,3.55,0,0,,無法直接判斷
kvdrive2026,"RTX 4090 ×1, 120GB DDR5",參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,RTX4090,1,165.2,131072,194.43,30632,25.6,假設 80%,7.0,假設 NVMe Gen4,37.97,10.38,0,0,0,63.29,27.12,17.31,7.42,0,0,,無法直接判斷
tutti2026,"H100 80GB ×1（型號未載明，以 SXM 計）, DRAM-HBM 50GB/s, 2× Solidigm PS1010 29GB/s",論文模型,Llama-3.1-8B,H100SXM,1,989.5,131072,32.46,30632,50.0,論文 p3「50 GB/s DRAM-HBM」,29.0,論文 p3 峰值 29 GB/s（2 顆）,12.38,7.18,0,0,0,20.64,8.84,11.97,5.13,0,0,,與硬體 κ 不一致、與含軟體路徑的 κ 一致
tutti2026,同上，SSD 用 LMCache-GDS 實測擷取頻寬,論文模型,Llama-3.1-8B,H100SXM,1,989.5,131072,32.46,30632,50.0,論文 p3,11.9,論文 p10 實測 11.9 GB/s,12.38,2.95,0,0,10944,20.64,8.84,4.91,2.11,0,0,,與硬體 κ 不一致、與含軟體路徑的 κ 一致
tutti2026,同上，SSD 用 Tutti 實測擷取頻寬,論文模型,Llama-3.1-8B,H100SXM,1,989.5,131072,32.46,30632,50.0,論文 p3,25.9,論文 p10 實測 25.9 GB/s,12.38,6.41,0,0,0,20.64,8.84,10.69,4.58,0,0,,與硬體 κ 不一致、與含軟體路徑的 κ 一致
lmcache2025,H100 ×1（GMI Cloud，型號未載明，以 SXM 計），LMCache CPU 載入 400 Gbps,論文模型,Llama-3.1-8B,H100SXM,1,989.5,131072,32.46,30632,50.0,論文 p13 表 5 實測 400 Gbps,1.875,論文 p12 遠端 15 Gbps,12.38,0.46,35335,101301,233235,20.64,8.84,0.77,0.33,8948,61721,,方向一致；定量需 X≈1.6–2.3
lmcache2025,同上，vLLM 原生 CPU offload 88 Gbps,論文模型,Llama-3.1-8B,H100SXM,1,989.5,131072,32.46,30632,11.0,論文 p13 表 5 實測 88 Gbps,1.875,論文 p12,2.72,0.46,35335,101301,233235,4.54,1.95,0.77,0.33,8948,61721,,方向一致；定量需 X≈1.6–2.3
lmcache2025,B200 ×1，遠端 32 Gbps（圖 15，模型未載明）,論文模型,Llama-3.1-8B,B200,1,2250.0,131072,14.28,30632,51.2,假設 80%,4.0,論文 p14 圖 15（32 Gbps）,5.58,0.44,39680,109993,250618,9.29,3.98,0.73,0.31,11555,67805,,方向一致；定量需 X≈1.6–2.3
lmcache2025,B200 ×1，遠端 64 Gbps,論文模型,Llama-3.1-8B,B200,1,2250.0,131072,14.28,30632,51.2,假設,8.0,論文 p14 圖 15,5.58,0.87,4524,39680,109993,9.29,3.98,1.45,0.62,0,18587,,方向一致；定量需 X≈1.6–2.3
lmcache2025,B200 ×1，遠端 128 Gbps,論文模型,Llama-3.1-8B,B200,1,2250.0,131072,14.28,30632,51.2,假設,16.0,論文 p14 圖 15,5.58,1.74,0,4524,39680,9.29,3.98,2.9,1.24,0,0,,方向一致；定量需 X≈1.6–2.3
leoam2025,"RTX 4090 ×1, 120GB, Intel SSD 實測 ~7 GB/s",論文模型,LongChat-7B/Llama-2-7B,RTX4090,1,165.2,524288,163.2,25711,25.6,假設 80%,7.0,論文 p9 實測 ~7 GB/s,7.97,2.18,0,0,21489,13.28,5.69,3.63,1.56,0,0,decode 期稀疏讀取,無法直接判斷
leoam2025,"RTX 4090 ×1, 120GB, Intel SSD 實測 ~7 GB/s",參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,RTX4090,1,165.2,131072,194.43,30632,25.6,假設 80%,7.0,論文 p9 實測 ~7 GB/s,37.97,10.38,0,0,0,63.29,27.12,17.31,7.42,0,0,decode 期稀疏讀取,無法直接判斷
mooncake2025,A800 SXM4 ×8/節點，RDMA 800 Gbps（遠端 DRAM 池）,論文模型,Llama-2-70B(dummy),A800,8,2496.0,327680,110.54,52627,80.0,假設 800 Gbps×80%（遠端 DRAM，非本機 PCIe）,,SSD 未量化,26.99,,,,,44.98,19.28,,,,,,一致
mooncake2025,A800 SXM4 ×8/節點，RDMA 800 Gbps（遠端 DRAM 池）,參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,A800,1,312.0,131072,102.95,30632,10.0,假設 800 Gbps×80%（遠端 DRAM，非本機 PCIe）,,SSD 未量化,7.85,,,,,13.09,5.61,,,,,,一致
cacheblend2025,"A40 ×1, 128GB RAM, 1TB NVMe 實測 4.8 GB/s",論文模型,Mistral-7B-v0.3,A40,1,149.7,131072,193.72,27656,25.6,假設 80%,4.8,論文 p10 實測 4.8 GB/s,37.84,7.09,0,0,0,63.06,27.03,11.82,5.07,0,0,,一致
cacheblend2025,"A40 ×1, 128GB RAM, 1TB NVMe 實測 4.8 GB/s",參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,A40,1,149.7,131072,214.56,30632,25.6,假設 80%,4.8,論文 p10 實測 4.8 GB/s,41.91,7.86,0,0,0,69.84,29.93,13.1,5.61,0,0,,一致
cacheblend2025,同上，Slower Disk 4 Gbps,論文模型,Mistral-7B-v0.3,A40,1,149.7,131072,193.72,27656,25.6,假設,0.5,論文 p12 圖（4 Gbps）,37.84,0.74,9768,47193,122043,63.06,27.03,1.23,0.53,0,24738,,一致
cacheblend2025,同上，Slower Disk 4 Gbps,參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,A40,1,149.7,131072,214.56,30632,25.6,假設,0.5,論文 p12 圖（4 Gbps）,41.91,0.82,6793,44218,119068,69.84,29.93,1.36,0.58,0,21763,,一致
cacheblend2025,A40 ×2（Llama-70B 8-bit，以 BF16 算力計）,論文模型,Llama-3.1-70B,A40,2,299.4,327680,942.55,53825,51.2,假設,4.8,論文 p10,147.27,13.81,0,0,0,245.46,105.2,23.01,9.86,0,0,,一致
kvpr2025,"A100 40GB ×1, PCIe4 x16",論文模型,OPT-6.7B,A100,1,312.0,524288,85.26,25367,25.6,論文 p6 32 GB/s ×80%,,無 SSD,4.16,,,,,6.94,2.97,,,,,decode 期；其「重算」是投影重算,定義不同，無法直接判斷
kvpr2025,"A100 40GB ×1, PCIe4 x16",參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,A100,1,312.0,131072,102.95,30632,25.6,論文 p6 32 GB/s ×80%,,無 SSD,20.11,,,,,33.51,14.36,,,,,decode 期；其「重算」是投影重算,定義不同，無法直接判斷
kvpr2025,"Quadro RTX 5000 ×1, PCIe4 x8 16 GB/s",論文模型,OPT-6.7B,QuadroRTX5000,1,89.2,524288,298.21,25367,12.8,論文 p13 16 GB/s×80%,,無 SSD,7.28,,,,,12.13,5.2,,,,,算力 89.2 TFLOPS 為論文所列,定義不同，無法直接判斷
kvpr2025,"Quadro RTX 5000 ×1, PCIe4 x8 16 GB/s",參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,QuadroRTX5000,1,89.2,131072,360.09,30632,12.8,論文 p13 16 GB/s×80%,,無 SSD,35.17,,,,,58.61,25.12,,,,,算力 89.2 TFLOPS 為論文所列,定義不同，無法直接判斷
orbitflow2026,"RTX A5000 ×1, PCIe3 x16, 384GB",論文模型,Llama-3.1-8B,A5000,1,111.1,131072,289.11,30632,12.8,假設 80%,,無 SSD,28.23,,,,,47.06,20.17,,,,,LLaMA3-8B；decode 期,"無法直接判斷（decode 期卸載、無 SSD）；κ_cpu≈28（A5000, PCIe3）～142（4×A6000, 70B）→ 不考慮重算合理"
orbitflow2026,"RTX A6000 ×4, PCIe4 x16, 256GB",論文模型,Llama-3.1-70B,A6000,4,619.2,327680,455.75,53825,102.4,假設,,無 SSD,142.42,,,,,237.37,101.73,,,,,LLaMA3-70B,"無法直接判斷（decode 期卸載、無 SSD）；κ_cpu≈28（A5000, PCIe3）～142（4×A6000, 70B）→ 不考慮重算合理"
orbitflow2026,"RTX A6000 ×4, PCIe4 x16, 256GB",參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,A6000,1,154.8,131072,207.49,30632,25.6,假設,,無 SSD,40.53,,,,,67.54,28.95,,,,,LLaMA3-70B,"無法直接判斷（decode 期卸載、無 SSD）；κ_cpu≈28（A5000, PCIe3）～142（4×A6000, 70B）→ 不考慮重算合理"
bottlenecks2026,H100 SXM5（κ 與 TP 數無關），PCIe5 規格,論文模型,Llama-3.1-70B,H100SXM,1,989.5,327680,285.19,53825,51.2,假設 80%,,無 SSD,44.56,,,,,74.27,31.83,,,,,,一致（且 κ_crit≡本表 κ_cpu(p=0)）
bottlenecks2026,H100 SXM5（κ 與 TP 數無關），PCIe5 規格,參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,H100SXM,1,989.5,131072,32.46,30632,51.2,假設 80%,,無 SSD,12.68,,,,,21.13,9.06,,,,,,一致（且 κ_crit≡本表 κ_cpu(p=0)）
bottlenecks2026,同上，PCIe 用論文實測 15 GB/s,論文模型,Llama-3.1-70B,H100SXM,1,989.5,327680,285.19,53825,15.0,論文 p8 實測 15 GB/s,,無 SSD,13.06,,,,,21.76,9.33,,,,,,一致（且 κ_crit≡本表 κ_cpu(p=0)）
bottlenecks2026,同上，PCIe 用論文實測 15 GB/s,參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,H100SXM,1,989.5,131072,32.46,30632,15.0,論文 p8 實測 15 GB/s,,無 SSD,3.71,,,,,6.19,2.65,,,,,,一致（且 κ_crit≡本表 κ_cpu(p=0)）
bottlenecks2026,同上,論文模型,Qwen3-235B-A22B,H100SXM,1,989.5,192512,89.7,14408,15.0,論文 p8 實測 15 GB/s,,無 SSD,6.99,,,,,11.65,4.99,,,,,,一致（且 κ_crit≡本表 κ_cpu(p=0)）
flexgen2023,"T4 16GB (GCP), 208GB DRAM, NVMe 讀 2 GB/s",論文模型,OPT-30B,T4,1,65.0,1376256,1843.69,43538,12.0,論文 p2 圖「12 GB/s」,2.0,論文 p7「about 2GB/s」,16.08,2.68,0,0,21462,26.79,11.48,4.47,1.91,0,0,,一致
flexgen2023,"T4 16GB (GCP), 208GB DRAM, NVMe 讀 2 GB/s",參考 Llama-3.1-8B（單卡）,Llama-3.1-8B,T4,1,65.0,131072,494.15,30632,12.0,論文 p2 圖「12 GB/s」,2.0,論文 p7「about 2GB/s」,45.24,7.54,0,0,0,75.4,32.32,12.57,5.39,0,0,,一致
flexgen2023,同上,論文模型,OPT-175B,T4,1,65.0,4718592,10769.23,74174,12.0,論文 p2,2.0,論文 p7,27.39,4.56,0,0,0,45.65,19.56,7.61,3.26,0,0,,一致
```

---

## 5. DeepSeek 推理系統的 KV 設計（查證；URL 見 §8 的 D 系列）

### 5.1 DeepSeek-V3/R1 Inference System Overview（Open Source Week Day 6；統計窗 UTC+8 2025-02-27 12:00 → 02-28 12:00）[D1]

* **PD 分離＋大規模 EP**：「As we have adopted prefill-decode disaggregation architecture」。
  * Prefill：「Routed Expert EP32, MLA/Shared Expert DP32」，每單元 4 節點。
  * Decode：「Routed Expert EP144, MLA/Shared Expert DP144」，每單元 18 節點。
* **KV 在 prefill 與 decode 之間怎麼傳：文字沒有說明。** 系統圖（"Diagram of DeepSeek's Online Inference System"）只畫了三件事：
  * Prefill Service 與「**External KVCache Storage (Optional)**」之間有雙向箭頭。
  * Decode Service 有一支箭頭指向該儲存。
  * Prefill Service 接到 Decode Load Balancer。
  至於 KV 是直接 GPU→GPU 傳，還是經外部儲存，**公開文件未寫**。
* **精度**：「matrix multiplications and dispatch transmissions adopt the FP8 format … while core MLA computations and combine transmissions use the BF16 format」。**沒有提 KV cache 的儲存精度。**
* **On-disk KV cache 命中（原文，UTC+8 2025-02-27 12:00 → 02-28 12:00）**：
  * 「Total input tokens: 608B, of which 342B tokens (56.3%) hit the on-disk KV cache.」
  * 「Total output tokens: 168B … the average kvcache length per output token was 4,989 tokens.」
  * 「Each H800 node delivers an average throughput of ~73.7k tokens/s input (including cache hits) during prefilling or ~14.8k tokens/s output during decoding.」
  * 尖峰 278 節點、平均 226.75 節點（8×H800／節點）；以 $2/GPU·hr 計日成本 $87,072。
  * 註腳的 R1 定價：「$0.14/M input tokens (cache hit), $0.55/M input tokens (cache miss)」。

### 5.2 DeepSeek API「Context Caching on Disk」[D2][D3][D4]

| 項目 | 2024-08-02 公告（[D2]） | 2026-09-24 擷取的現行文件（[D3]） |
|---|---|---|
| 存在哪 | 「caches content that is expected to be reused on a distributed disk array」 | 「Each user request will trigger the construction of a hard disk cache」 |
| 命中條件 | 「only requests with identical prefixes (starting from the 0th token) will be considered duplicates. Partial matches in the middle of the input will not trigger a cache hit.」→ **只認從第 0 個 token 起的相同前綴**；與先前請求重疊的前綴部分可命中（儲存單位 64 token，見下一列） | 「Due to the Sliding Window Attention mechanism … Each cached prefix is an independent, complete unit. A subsequent request can only hit the cache if it fully matches a cache prefix unit.」→ **改為只能完整命中某個已持久化的「cache prefix unit」** |
| 持久化的時機／單位 | 「The cache system uses 64 tokens as a storage unit; content less than 64 tokens will not be cached.」 | 三種：①每個請求在「end position of the user input」與「end position of the model output」各產生一個 unit；②偵測到跨請求共同前綴時另存一個 unit；③長輸入／長輸出「at fixed token intervals」切 unit（**間隔未公開**）。現行文件**不再提 64 token** |
| 建立延遲 | — | 「Cache construction takes seconds.」 |
| 清除 | 「Unused cache entries are automatically cleared, typically within a few hours to days.」 | 「Once the cache is no longer in use, it will be automatically cleared, usually within a few hours to a few days.」 |
| 延遲效果 | 「For a 128K prompt with high reference, the first token latency is cut from 13s to just 500ms.」 | — |
| 為什麼能用碟 | 「made possible by the MLA architecture in DeepSeek V2 … significantly reducing the size of the context KV cache, enabling efficient storage on low-cost disks」 | — |
| 命中計價 | 命中 $0.014／M，未命中 $0.14／M → **10%** | [D4]（2026-09-24）：deepseek-flash（V4.1-Flash）命中 $0.003／$0.006（離峰／尖峰），未命中 $0.15／$0.3 → **2%**；deepseek-v4-pro 命中 $0.022／$0.044、未命中 $0.66／$1.32 → **3.3%** |

另一個時點是 [D1] 的 R1 定價（2025-02）：命中 $0.14、未命中 $0.55 → **25%**。**命中比例隨時期在 2%–25% 之間**，這是商業定價，不是成本量測。

### 5.3 3FS（Fire-Flyer File System）[D5][D6]

* **用途**（README）：「KVCache for Inference: Provides a cost-effective alternative to DRAM-based caching, offering high throughput and significantly larger capacity.」
* **KV 怎麼存：公開文件沒有說 KVCache 的 key／檔案佈局。** 3FS 本身是檔案系統：
  * metadata 存在 FoundationDB（「3FS stores all metadata as key-value pairs in FoundationDB」）。
  * 資料「split into equally sized chunks, which are replicated over multiple SSDs」，用 CRAQ 複製（design notes）。
  * 所以「KV 是否以 key-value 形式存」**查無**；能確定的只有底層是 chunk 化的檔案。
* **讀取頻寬**：
  * KVCache 客戶端（「1×400Gbps NIC/node」）「peak throughput reaching up to 40 GiB/s」。
  * 圖上另有一條「Average Read」，我讀圖約 2–4 GiB/s（**圖讀值，約略**），peak 曲線約 30–42 GiB/s。
  * 大規模讀壓測：「180 storage nodes, each equipped with 2×200Gbps InfiniBand NICs and sixteen 14TiB NVMe SSDs … aggregate read throughput reached approximately 6.6 TiB/s」。
* **GC／刪除**：README 圖二是「the IOPS of removing ops from garbage collection (GC)」。我讀圖是週期性尖峰，約 1.0–1.4 MIOPS、約每 70–80 秒一次（**圖讀值**）。**刪除的觸發條件、TTL 未公開。**
* **網路配置**（ISCA'25 [D9] p9）：「each node has a 400 Gbps Ethernet RoCE NIC connected to a separate storage network plane for accessing the 3FS」。

### 5.4 FlashMLA [D7][D8]

* **初版（2025-02-21，commit 414a2f3）**：「BF16」「**Paged kvcache with block size of 64**」。vLLM 對 V3.2 也「only support block size 64」[D11]。
* **FP8 KV（V3.2，只在 sparse 路徑）**：「a page block is `page_block_size` token-major rows of **656 Bytes** each」。組成：512 個 float8_e4m3（quantized NoPE）＋4 個 float32 scale（每 128 值一個）＋64 個 bfloat16（RoPE，「not quantized for accuracy」）。kernel 先反量化成 BF16 再算。dense decoding kernel 只讀 BF16／FP16。
* **V4／V4.1**（README 2026-09-10 條目，**2026 年資料**）：每 token 584 B（V4）、528 B（V4.1，RoPE 也量化成 FP8）、288 B（V4.1 fp4，只能用在 `extra_k_cache`）。原文：「In practice we expect the sliding window (SWA) kv cache to be in FP8 and the compress attention (CA) kv cache to be in FP4.」

### 5.5 DeepSeek Sparse Attention（V3.2-Exp）[D10][D11]

* **機制**（技術報告）：
  * lightning indexer 對每個前序 token 算 index score。「Given that the lightning indexer has a small number of heads and can be implemented in FP8, its computational efficiency is remarkable.」
  * 再「retrieves only the key-value entries {c_s} corresponding to the top-k index scores」。
  * 稀疏訓練階段「select **2048** key-value tokens for each query token」（p3）。
  * DSA 建在 MLA 的 **MQA 模式**上（每個 latent 被所有 query head 共用）（p2）。
  * 複雜度：主注意力從 O(L²) 降到 O(Lk)，「Although the lightning indexer still has a complexity of O(L²), it requires much less computation」（p4）。
* **config**：`index_n_heads`＝64、`index_head_dim`＝128、`index_topk`＝2048 [D12]。
* **改變了 KV 的哪一部分**：多了一份 **indexer K cache**。vLLM：「For each token, there is another K cache used by the indexer」「allocates separate buffers」，且「stored on a per-block basis … The first `block_size * head_dim` entries contain the value, the rest contain the scaling factor」[D11]。MLA 的 latent KV 在 FP8 時仍是 656 B/層/token。
  * 我們的算術：indexer key 每層每 token 128 個值；若為 FP8＋一個 fp32 scale 即 132 B。**scale 大小原文未寫明**。
* **是否仍需完整 KV 在 HBM**：公開文件**沒有**描述把未入選的 KV 卸載。依機制，top-k 由每個 query 動態決定，任何 token 都可能被選中，所以所有 latent KV 與 indexer key 都必須保留、可存取。這是我們的推論，文件未明寫。
* **對放置／卸載的意義**（我們的算術）：
  * 每層每位置的主注意力 FLOPs 在 k＝2048 封頂。indexer 的斜率是 2·64·128＝16,384 FLOPs／位置／層（FP8）。
  * 以 2N＝74 GFLOP 計，indexer 讓成本翻倍的位置 D≈74K。MHA 模式的 D 是 14.8K，約 **5 倍的斜率下降**；FP8 再減半。
  * **「重算成本隨位置線性上升」的前提被削弱**，P* 的位置判準在這類模型上幾乎失效。
  * 另一方面，每步的主注意力只讀 2,048 個 latent entry，但 indexer 仍要掃過全部 token 的 indexer key。所以理論上可以讓 indexer key 常駐 HBM、latent KV 從低階稀疏抓取（Quest／ShadowKV／KVDrive 類）。**DeepSeek 自己沒說他們這樣做。**
* **V4**（vLLM blog 2026-04-24 [D13]，2026 年資料）：
  * 「a sliding window of size 128」。
  * 壓縮注意力 c4a（「roughly 1/4」）與 c128a（「roughly 1/128」），DSA top-k 512／8192。
  * 「With bf16 KV cache, DeepSeek V4 only has 9.62 GiB KV cache per sequence at 1M context. That is about 8.7x smaller than the 83.9 GiB estimate for a 61-layer DeepSeek V3.2-style stack.」vLLM 用「fp4 for the indexer cache and fp8 for the attention cache」。
  * 我們的算術：9.62 GiB/2²⁰ token ≈ 9.9 KB/token，是 Llama-3.1-8B 的 1/13。

### 5.6 ISCA'25「Insights into DeepSeek-V3」表 1（BF16）[D9]

| Model | KV Cache Per Token | Multiplier |
|---|---|---|
| DeepSeek-V3 (MLA) | **70.272 KB** | 1× |
| Qwen-2.5 72B (GQA) | 327.680 KB | 4.66× |
| LLaMA-3.1 405B (GQA) | 516.096 KB | 7.28× |

（p4。我們由 config 核對：(512＋64)×2 B×61 層＝70,272 B；V2 為 60 層＝69,120 B [D12]。Bottlenecks 表 3 也用 70 KB。）

同文 §4.5.1（p8）：「transferring KV cache data from CPU memory to GPU can consume tens of GB/s, saturating PCIe bandwidth. If the GPU simultaneously uses IB for EP communication, this contention … can degrade overall performance」。這是**生產端承認 KV 搬運與 EP 通訊搶 PCIe** 的直接證據，也就是 §6 說的「負載／並發」因子。

### 5.7 逐出／TTL 策略

公開資料**只有**兩句 API 文件：「usually within a few hours to a few days」「Unused cache entries are automatically cleared」，外加 3FS README 的 GC IOPS 圖。**沒有**公開的逐出演算法（LRU/LFU/成本感知）、容量配額或 TTL 精確值。**寫「沒有」。**

### 5.8 MLA 讓 κ 往哪移、對 P* 的意義、生產佐證力（我們的算術）

| 情境（H800：BF16 dense 989.5／FP8 dense 1,979 TFLOPS，MFU 0.5） | t_rc(0) µs | κ_cpu（51.2 GB/s） | κ_ssd（NVMe 7） | κ_ssd（SATA 0.55） | BW*（κ_ssd＝1） |
|---|---|---|---|---|---|
| Llama-3.1-8B，BF16 算力 | 32.5 | 12.7 | 1.73 | 0.14 | 4.04 GB/s |
| DeepSeek-V3，BF16 KV，BF16 算力 | 149.6 | 109 | 14.9 | 1.17 | **0.47 GB/s** |
| DeepSeek-V3，BF16 KV，FP8 算力 | 74.8 | 54.5 | 7.45 | 0.59 | 0.94 GB/s |
| DeepSeek-V3，FP8 KV（656 B/層），FP8 算力 | 74.8 | 95.7 | 13.1 | 1.03 | 0.54 GB/s |
| 整節點 8×H800（FP8 算力）＋3FS 單客戶端峰值 40 GiB/s | 9.35／token·節點 | — | 3FS：**5.7**（BF16 KV）／**10.0**（FP8 KV） | — | 7.5／4.3 GB/s／節點 |

* **方向**：MLA 把 2N/B_kv 從 Llama-8B 的 0.1225 提高到 1.05 MFLOP/B（**8.6 倍**），κ 同比上升，P* 往 0 移。只要碟 ≥0.5–1 GB/s，任何位置都是載回贏。
* 對照 Bottlenecks 表 3 的 κ_M：DeepSeek-V3 1.06 vs Qwen3-235B 0.23（4.6 倍）。
* MLA 的 D＝14.8K（prefill MHA 模式；decode MQA 模式為 4.4K）比 Llama-8B 的 30.6K 短，位置越後面重算越貴，也往「載回」推。
* **生產數字的一致性檢查**（我們的算術，非 DeepSeek 的陳述）：
  * 73.7k input token/s/節點 × 56.3% 命中 ≈ 41.5k token/s 由快取供應。以 BF16 KV 70,272 B 計 ≈ **2.9 GB/s／節點**（2.7 GiB/s），落在 3FS 圖中「Average Read」約 2–4 GiB/s 的帶內（**圖讀值**），且遠低於每節點 400 Gbps 儲存 NIC 與 40 GiB/s 峰值。
  * 反推實際重算：32.2k token/s/節點（未命中）× 74 GFLOP ≈ 2.4 PFLOP/s，相當於 FP8 峰值的 15%（BF16 的 30%）。
  * 若用這個「實際」重算成本（31 µs/token·節點），對 3FS 峰值的 κ 約 19。
* **佐證力**：
  * **強**：「SSD／分散式儲存階在生產中存在且划算」。56.3% 的 token 來自碟；命中定價只有 2–25%；13 s → 500 ms。
  * **有限**：它是 **MLA＋大 batch EP＋專用 3FS 叢集＋RDMA 儲存網路**的組合。κ_ssd 在這個組合下 ≥5，不能外推到 GQA 模型配單顆本地碟（例如平台 B 的 Llama-8B 實測 κ_ssd＝0.53）。
  * **DeepSeek 的決策空間裡「重算 vs 載回」幾乎不需要判**：命中就載，未命中才算；只剩「能不能命中」（prefix unit 粒度）這一題。

### 5.9 對 Tiara 三個決策的意義

| Tiara 決策 | DeepSeek 的做法（證據） | 代表什麼 |
|---|---|---|
| **幾 bit** | V3.2 的 KV 在 sparse 路徑以 FP8 存（656 B/層），RoPE 64 維刻意留 BF16「for accuracy」[D7]；V4.1 的 SWA KV 用 FP8、壓縮注意力 KV 用 FP4 [D7]；vLLM 對 V4 用 FP4 indexer／FP8 attention cache [D13] | **生產證據**：KV 精度是「依 KV 成分／注意力類型」靜態決定的（RoPE vs NoPE、SWA vs 壓縮、indexer vs 主 KV），**不是**依 block 重要度或 κ 動態決定。對 Tiara「每個 block 動態選 bit」是**反例**：生產端選擇了結構化、靜態、依成分的混合精度 |
| **放哪一階** | on-disk KV（3FS，分散式 NVMe＋RDMA）服務 56.3% 的輸入 token [D1][D5]；系統圖把外部 KV 儲存標為「Optional」[D1]；API 說「distributed disk array」[D2]；ISCA'25 §4.5.1 提到「transferring KV cache data from CPU memory to GPU」[D9] | **生產證據**：在 κ 很大（MLA）且有專用儲存網路時，公開描述的放置是 **HBM＋分散式碟**。CPU 記憶體出現在傳輸路徑上（ISCA'25），但**是否作為獨立的快取階，公開文件沒有說明**。與平台 B 的發現一致：階間差距被壓縮時，簡單兩階已接近最佳（CLAIM_EVIDENCE C1）。對 Tiara 的六態動作空間是**部分反例**：中間階不一定需要 |
| **重算 vs 載回** | 命中就載（「bypassing the need for recomputation」[D2]）；命中條件是前綴（2024）／完整 prefix unit（現行）[D2][D3]；無逐出策略公開 | **生產證據**：MLA 的 κ≫1 使「載回永遠贏」，所以 DeepSeek **不做**逐 block 的重算判斷。真正的問題變成「**命中粒度**」：現行 SWA 設計只允許完整命中 unit，**無法命中的部分只能重算**。Tiara 的 P*（位置判準）在 MLA＋DSA 模型上意義很小（D 大、κ 大）；它主要適用於 GQA/MHA、κ≈1 的平台（例如平台 A/B 的 Llama-8B，κ_ssd 0.5–0.9） |

---

## 6. 結論

### 6.1 「用 κ 把大家設備列出來再選對應 SOTA」站得住嗎？

**當一階地圖：站得住。** 18 篇裡：
* 只要把各自的硬體代進去，11 篇的結論方向就能解釋。
* Cake 的表 3（差 12–20%）、CachedAttention 的 360/192 ms（2.08 vs 1.9）、HCache 的 28% 降速（公式 23%），都能用同一條公式在約 10–20% 內重現（§1.1）。
* 使用者的直覺也有一半在表上成立：伺服器卡的 κ_ssd 由碟的等級決定，SATA 與 Gen5 NVMe 可以差 25 倍。

**但另一半方向相反**：邊緣卡的 κ 最大，硬體**不支持**「邊緣上長請求重算較好」。MTDS 的結論要用容量與軟體來解釋。

**要補的五件事**（依證據強度排序；前三項已有實測）：

1. **軟體路徑 X（獨立因子，不是硬體的一部分）**
   * 證據：Tutti（約 13×以上）、Bottlenecks（6.5×）、LMCache（同硬體 4.5×）、平台 A（CPU 7.2×、SSD 1.3–18×）、平台 B（CPU 13.4×、SSD 2.6×、同 PCIe 跨模型 17.1×）。
   * 平台 B 的機制已查到每筆描述符的固定成本，所以 X 可以寫成 **X ≈ 1 + (n_desc × c_desc) / (bytes / BW_hw)**（我們的算術形式；n_desc 取決於模型的層數與 KV 佈局）。
   * **這一項讓「κ 跨硬體 32 倍」必須改寫**：同一張卡上跨模型的 κ_cpu 差 9.5 倍，跨平台實測只差 6 倍。變異的主因是「(模型, 實作路徑) 有多會用硬體」，不只是硬體。
2. **位置與 ctx**
   * κ(p)＝κ(0)·(1＋p/D)。實測 D 比公式短 1.6–1.9 倍（3090 19.1K、MI300X 16.1K）。
   * 任何 κ 都要標明模型與 ctx（CLAIM_EVIDENCE B1 也是這個結論）。
3. **負載／機會成本**
   * BiDAW 的單請求 κ_ssd≈0.3，照理應該重算；它在 GPU 滿載時仍然贏，因為重算吃的是被搶的 GPU，載回吃的是閒置的 I/O。
   * DeepSeek ISCA'25 也承認 KV 搬運與 EP 通訊搶 PCIe。
   * 建議形式：比較「**GPU-秒**」而非延遲。例如 κ_eff＝t_rc·(1+ρ_GPU)／[X·t_io·(1+ρ_IO)]，ρ 為各資源的佇列負載（**建議形式，未驗證**）。
   * 注意 CLAUDE.md 規定**平台 A 不做機會成本結論**。
4. **可預取性**：CachedAttention 把碟讀移出關鍵路徑，κ_ssd≤0.4 的碟照樣有用。κ 只該算「在關鍵路徑上的那一段」。
5. **模型架構與容量**
   * MLA（κ×8.6）、MoE（active 小 → κ 小，Qwen3-30B-A3B D＝8.5K）。
   * DSA／壓縮注意力（D×5～10，位置判準變弱）。
   * HCache/BiDAW 的 hidden-state 動作只對 MHA 有利（BiDAW p10、OrbitFlow p12）。
   * 容量懸崖（MTDS 的 5 GB VRAM buffer；平台 B 的 `/dev/shm` 178.8 GiB）會讓「載回」在容量不足時失效，這**不是頻寬現象**。

**所以選 SOTA 的鍵不是「GPU 型號」，而是 (κ_hw, X, 模型, ctx 分佈, 負載) 五元組。** 上表的硬體欄可以直接用；X 欄目前只有 Tutti、Bottlenecks、LMCache、Strata 與我們兩個平台有數字，其餘論文都沒揭露軟體路徑的有效頻寬。

### 6.2 能不能回答教授的問題「每一格都有人做得更深，他們在哪裡做不到」？

**能回答「在哪裡做不到」的那一半**：每篇的固定規則在哪個 κ 區間成立、在哪裡失效，多半有論文自己的數字可引。

| 論文 | 固定規則 | 規則成立的區間（證據） | 規則失效／沒涵蓋的區間（證據） | 缺的維度 |
|---|---|---|---|---|
| Cake | 永遠雙向並行 | 算力與 I/O 平衡時（κ_whole 約 0.5–2；表 3 的 teal 格＝對兩個 baseline 都 ≥1.5×，p6） | κ≪1 或 ≫1：7 Gbps＋短序列比 compute-only 慢（p7）；I/O 是模擬（p5） | 軟體路徑、SSD 實體、精度 |
| Strata | CPU 階、I/O 機制＋排程；不重算 | PCIe5／C2C 的 CPU 階（p5、p13） | 自家 H200 的碟 κ_ssd≈0.27（P*≈85K）→ 直接不用碟（p13），沒有建模「碟 vs 重算」 | 位置相依的碟決策、精度 |
| MTDS | 依歷史延遲選 載回／部分載回／重算 | 容量不足、軟體慢的邊緣配置 | 任何有效載回 ≥1 GB/s 的配置（A10 κ_cpu≈50）：「越長越重算」與物理相反 | 位置、軟體路徑的量化 |
| BiDAW／CachedAttention | 兩階（DRAM＋碟）＋排程／預取 | GPU 滿載、可預取（p5；p4） | 閒置 GPU 上單請求 κ_ssd≈0.3–0.4 → 重算較快 | 負載項、精度 |
| AdaptCache／EvicPress | 壓縮率＋放置聯合選 | 碟慢（κ_ssd≈0.5–0.8） | 重算只當靜態 baseline；碟頻寬固定；「bandwidths … similar」時優勢消失（EvicPress p12） | 位置相依的重算成本 |
| Tutti | GPU-centric SSD I/O，永遠載回 | 修好軟體路徑後 κ_ssd≈6 | 不處理重算、精度 | 決策層 |
| LMCache | 載回（建議「should be adaptive」但未實作） | 高頻寬網路 | 32 Gbps＜256K 時輸給 prefill（p14） | 自動的交叉判斷 |
| Bottlenecks | 只做特性分析 | — | 無 policy | — |
| HCache／KVPR | activation→KV 第三動作 | MHA、IO 與算力平衡 | GQA 上 activation 不再輕（OrbitFlow p12；BiDAW p10） | — |
| KVDrive／LeoAM／OrbitFlow | decode 期稀疏讀取／卸載 | 長上下文 decode | 不涉及 prefill 的重算 vs 載回 | — |
| DeepSeek（生產） | 命中就載，否則算；FP8／FP4 靜態精度 | MLA＋3FS（κ_ssd≥5） | 公開資料無逐出策略；SWA 後只能完整命中 unit | —（對 GQA、單碟平台不適用） |

**不能回答的那一半**：「做得更深」要成立，必須有 Tiara 在那些失效區間實際勝出的**量測**。這張表只是算術＋論文數字，**沒有任何一格是 Tiara 贏的證據**。而且專案自己的平台 B 結果對 Tiara 不利：
* 學習式放置在平台 B 輸給最佳 baseline（CLAIM_EVIDENCE C1）。
* INT4 單獨就吃掉 oracle headroom 的 91.8%（EXPERIMENTS 實驗 1）。

所以這張表比較適合當「**問題定義**」的證據：沒有一篇同時把**位置相依的重算成本、實測的軟體路徑 X、負載**放進同一個決策；而且各篇結論的分歧可以被這些缺的維度解釋。它**不適合**當「Tiara 比較好」的證據。最有支撐的新敘事是：「**多階 KV 的有效成本是 (實作路徑 × 模型) 的性質**」（平台 B 實驗 2 已有機制與三重確認）。

---

## 7. 限制與未查證

* **非官方規格**：H20、L20 沒有 NVIDIA 公開規格表；H800、A30、Quadro RTX 5000 用論文所列數字；GH200 算力、B200 單卡 host link **未查證**；A800 SXM 的 HBM 頻寬是推定。
* **型號不明**：Cake、EvicPress、Tutti、LMCache 的 H100 是 SXM 還是 PCIe，論文未寫（κ 相差 1.31 倍）。MTDS、KVDrive、EvicPress、CachedAttention 的 SSD 型號或頻寬未揭露，MTDS／KVDrive 列用的是**假設值**（SATA 0.55、NVMe Gen4 7 GB/s）。
* **MFU 不確定性約 ±2 倍**：平台 A 實測線性部分的 MFU 為 0.9，平台 B 只有 0.36，都不是 0.5。κ 在 0.5–2 之間的判定只看方向。
* **TP 聚合**：假設各卡獨立 x16，實際 SXM 平台常經 PCIe switch 共享（MTDS p7 也提到），多卡列的 κ_cpu 可能高估。
* **位置模型**：只有線性注意力項，沒有 FlashAttention 的實際效率曲線與 chunk 邊界效應；實測 D 比公式短 1.6–1.9 倍。
* **平台 B 的 SSD 階**是容器 overlay（Broadcom MegaRAID，不是 NVMe，暫態），真實部署會不同（ADVISOR_REPLY §1.1 已要求寫進 threats to validity）。
* **平台 A 的 NVMe P*（10,851）**：RUNLOG 發現 11 記錄 NVMe 那次量測有累計 733 s 的 lookup 延遲、標為待查；10,851 是否出自同一批量測未能確認，在釐清前不宜拿來比較裝置。
* **DeepSeek V4／V4.1 的內容**（FlashMLA README 2026-09-10、vLLM blog 2026-04-24、API 定價頁 2026-09-24）是 2026 年的線上資料，逐字引用自擷取當下的頁面；模型細節未另行查證。
* **圖讀值**：3FS 的平均讀與 GC 週期是讀圖估計，已標註。

---

## 8. 來源 URL

**GPU**
* [G1] NVIDIA RTX Blackwell GPU Architecture whitepaper（表 3：3090／4090／5090）— https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
* [G2] NVIDIA Ampere GA102 GPU Architecture whitepaper v2（表 3：A6000/A40；表 9：3090；p9 PCIe Gen4）— https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
* [G3] NVIDIA Ada GPU Architecture whitepaper（附錄 A 表 2：4090）— https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
* [G4] NVIDIA A10 datasheet — https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/datasheet-new/nvidia-a10-datasheet.pdf（另：A10 product brief PB-10415 https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/A10-Product-Brief.pdf）
* [G5] NVIDIA A40 datasheet — https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf
* [G6] NVIDIA RTX A5000 datasheet — https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-a5000-datasheet.pdf
* [G7] NVIDIA T4 datasheet — https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
* [G8] L20（第三方彙整，非官方）— https://flopper.io/gpu/nvidia-l20-48gb/spec-sheet
* [G9] NVIDIA A100 datasheet — https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
* [G10] Lenovo Press LP1813 ThinkSystem NVIDIA A800 PCIe — https://lenovopress.lenovo.com/lp1813-thinksystem-nvidia-a800-pcie-gpu
* [G11] Tencent Cloud HCC 實例規格（A800 NVLink 80GB）— https://www.tencentcloud.com/document/product/1236/62812
* [G12] NVIDIA H100 產品頁 — https://www.nvidia.com/en-us/data-center/h100/
* [G13] NVIDIA H100 Tensor Core GPU datasheet（經銷商鏡像）— https://www.megware.com/fileadmin/user_upload/LandingPage%20NVIDIA/nvidia-h100-datasheet.pdf
* [G14] NVIDIA H100 PCIe product brief PB-11133 — https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf
* [G15] NVIDIA H200 產品頁 — https://www.nvidia.com/en-us/data-center/h200/
* [G16] H20（媒體轉述，非官方）— https://wccftech.com/nvidia-china-compliant-h20-gpu-41-percent-fewer-cores-lower-performance-vs-top-hopper-h100/ ；https://viperatech.com/product/nvidia-hgx-h20
* [G17] NVIDIA Technical Blog: Grace Hopper Superchip Architecture In-Depth — https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/
* [G18] NVIDIA GH200 產品頁 — https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/
* [G19] NVIDIA HGX（B200 規格，「Dense is ½ sparse」）— https://www.nvidia.com/en-us/data-center/hgx/
* [G20] NVIDIA DGX B200 — https://www.nvidia.com/en-us/data-center/dgx-b200/
* [G21] AMD Instinct MI300X data sheet — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf

**SSD**
* [S1] Samsung PM9A3 U.2 3.84TB（MZ-QL23T800）— https://www.samsung.com/us/business/memory-storage/nvme-ssd/pm9a3-nvme-u-2-ssd-3-8tb-sku-mz-ql23t800/
* [S2] Samsung PM9A3 product brief — https://download.semiconductor.samsung.com/resources/brochure/Samsung%20PM9A3%20NVMe%20PCIe%20SSD.pdf
* [S3] Samsung SSD 870 QVO Data Sheet Rev1.1 — https://download.semiconductor.samsung.com/resources/data-sheet/Samsung_SSD_870_QVO_Data_Sheet_Rev1.1.pdf
* [S4] Crucial P3 B2B product flyer — https://www.asipartner.com/wp-content/uploads/2022/07/crucial-p3-b2b-flyer.pdf
* [S5] Crucial P3 consumer product flyer（含 TBW；鏡像）— https://gzhls.at/blob/ldb/2/8/1/d/1801eb7b4575ea1779b9bfa8f2699fc5f7f2.pdf
* [S6] StorageReview: Intel P5510 review（規格表）— https://www.storagereview.com/review/intel-p5510-nvme-enterprise-ssd-review
* [S7] Solidigm D7-PS1010 產品頁 — https://www.solidigm.com/products/data-center/d7/ps1010.html
* [S8] Solidigm D7-PS1010（standard endurance）— https://www.solidigm.com/archive-v2/products/data-center/d7/ps1010.html
* [S9] Samsung 980 PRO 1TB — https://www.samsung.com/us/computing/memory-storage/solid-state-drives/980-pro-pcie-4-0-nvme-ssd-1tb-mz-v8p1t0b-am/

**DeepSeek**
* [D1] Day 6: DeepSeek-V3/R1 Inference System Overview — https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
* [D2] DeepSeek API News 2024/08/02: Context Caching on Disk — https://api-docs.deepseek.com/news/news0802
* [D3] DeepSeek API Guide: Context Caching（2026-09-24 擷取）— https://api-docs.deepseek.com/guides/kv_cache
* [D4] DeepSeek API Models & Pricing（2026-09-24 擷取）— https://api-docs.deepseek.com/quick_start/pricing
* [D5] 3FS README — https://github.com/deepseek-ai/3FS
* [D6] 3FS design notes — https://github.com/deepseek-ai/3FS/blob/main/docs/design_notes.md
* [D7] FlashMLA README（現行）— https://github.com/deepseek-ai/FlashMLA
* [D8] FlashMLA 初版 README（commit 414a2f3，2025-02-21）— https://github.com/deepseek-ai/FlashMLA/blob/414a2f3eedeb5ad3c4a6e89d8641e059519cacc9/README.md
* [D9] Zhao et al., Insights into DeepSeek-V3（ISCA'25）— https://arxiv.org/abs/2505.09343
* [D10] DeepSeek-V3.2-Exp 技術報告 — https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf
* [D11] vLLM blog: DeepSeek-V3.2-Exp in vLLM — https://vllm.ai/blog/2025-09-29-deepseek-v3-2
* [D12] HF config：DeepSeek-V3 https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json ；V3.2-Exp https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/config.json ；V2 https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/config.json
* [D13] vLLM blog: DeepSeek V4 in vLLM（2026-04-24）— https://vllm.ai/blog/2026-04-24-deepseek-v4

**模型 config**：見 §2.3 表內連結。**論文**：本地文字檔 `scratchpad/papers_txt/<key>.txt`。**專案實測**：`results/RUNLOG.md`、`results/m4_oracle/*.json`（平台 A）；`scratchpad/remote/results/RUNLOG_MI300X.md`、`results/m4_oracle_mi300x/cost_model_b-*.json`、`results/CLAIM_EVIDENCE_MI300X.md`、`ADVISOR_REPLY_20260919.md`、`EXPERIMENTS_20260919.md`（平台 B）。
