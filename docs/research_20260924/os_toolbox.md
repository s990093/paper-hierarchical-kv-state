# Tiara 的 OS／儲存／快取理論工具箱（文獻深度查證版）

> **版本**：2026-09-24（已納入協調者 9/24 轉來的 9/19–9/21 平台 B 新實測）
> **目的**：找出經典 OS／儲存／快取／排程／線上演算法理論中，Tiara 可以拿來 (1) 用一條簡單定律講故事、(2) 在實驗前先建理論模型、(3) 設計線上策略、(4) 給 Oracle 上下界、(5) 解釋或威脅既有發現的工具；每條都查到原始出處。
> **查證方式**：每條至少打開一個一手頁面（USENIX／ACM／IEEE／INFORMS／Springer／arXiv／作者 PDF）。其中約 40 篇另下載 PDF、用 PyMuPDF 抽全文逐字核對（存在同一 scratchpad 的 `refs_txt/`）。「是否已有 KV 論文用過」同時用 web 搜尋與本地 35 篇 KV 論文全文 grep（`papers_txt/`）。
> **標記約定**
> - 【原文】＝論文原文所述。後面註「全文核對」＝我抽了 PDF 全文逐字看過；「摘要核對」＝只核到摘要／出版頁。
> - （我們的判讀）＝本文的推論，不是任何論文說的。
> - （我們的推導，非文獻）＝本文自己推的關係式。
> - 「未查證」＝查不到或無法確認；「本次搜尋未找到」≠「不存在」（CLAUDE.md 規則 7）。
> - 🔴 **本文沒有任何新量測。** Tiara 的數字只來自 §0；其他數字要嘛是論文原文數字（附出處），要嘛是「由 §0 數字做的算術」（一律標「算術，非實測」）。

---

## 目錄

- §0 已量測事實 F1–F11（只引用，不改）
- §1 一覽表
- §2 逐條查證（A 定律 → O 排隊），每條 (a) 定義 (b) 引用 (c) 對應決策與解釋／威脅 (d) KV 先例 (e) 便宜實驗 (f) 新穎性風險
- §3 回應協調者的三個新問題（替代關係、Little's Law 實例、learning-augmented 證據）
- §4 Tiara 的一句話定律候選（我們的推導，非文獻）
- §5 對 Tiara 最有用的前 8 項
- §6 「先做理論模型」的最小路線（全部是提案，未執行）
- §7 新威脅文獻與「護城河」修正
- §8 查證紀錄與未查證清單

---

## §0 已量測事實（只引用，不改；以下簡稱 F1–F11）

**任務給定（平台 A 為主）**

- **F1** 重算成本對固定大小 chunk 隨「前序已快取 token 數 P」線性成長：Llama-3.1-8B BF16、2048-token chunk：`513 ms + 26.9 ms × (P/1000)`。
- **F2** SSD 取回成本與位置無關 → SSD 與 DROP 有交叉點 P*（Qwen2.5-7B AWQ + NVMe：37,717 token；Llama BF16 + NVMe：10,851）。vLLM `OffloadingConnector` 的 SSD 階約 65% 成本是軟體層（lookup、cascade、promotion），不是 I/O。
- **F3** 模擬器加「降成 INT8」精度階：降級 780,108 次，總時間只變 +0.004%（逐出照 LRU 排序，被降級的 block 仍是最冷的，下一次就被搬走）；改成「剩餘價值÷位元組」排序後同設定 +15.40% → −18.00%。
- **F4** 學習式預測器在沒有資訊時比 LRU 差 2.7–5.6 倍；AUC 0.9995 但逐出順序幾乎沒學到（AUC 問會不會再用，逐出需要誰先被用）。
- **F5** decode 佔端到端時間 48.8–75%，放置只能影響 prefill。
- **F6** Oracle 是貪婪解，不保證最優。
- **F7** Strata 已用 Little's Law（§3.1：C = λ·L、X = λ·S → X = C·S/L，論證「放大每次傳輸 S」），也處理 delay hit 與 bundle hit。

**協調者 9/24 補充（出處：`scratchpad/remote/EXPERIMENTS_20260919.md`、`ADVISOR_REPLY_20260919.md`；我只引用、未重跑）**

- **F8**（平台 B，MI300X）全域改 INT4 KV（`int4_per_token_head`）+ 原廠 LRU 就拿到 oracle headroom 的 91.8%（toolagent）／103.9%（conversation）；切 INT4 之後剩餘 headroom 2.02%／3.52%；兩個 oracle 幾乎一樣（toolagent：1,370,923 vs 1,371,969 ms）。INT4 容量倍數 3.77×（M1）。原文結論：「量化與放置在延遲上是替代品，不是互補品」。
- **F9** vLLM `swap_blocks_batch` 在 MI300X 的有效頻寬由「每筆描述符固定成本」決定（原文推算 12.9 µs @32 KiB／16.3 µs @1.5 MiB，與 payload 無關）；bulk copy 57.4 GB/s；微基準重現端到端實測（32 KiB：2.4 vs 2.27 GB/s；64 KiB：4.7 vs 4.47；1.5 MiB：38.9 vs 38.28）；跨模型 17.1× 差異由描述符粒度解釋。
- **F10** 學習式放置在 MI300X 輸給 `tier_fs` 9.7–47.3%（同程式碼在 3090 為 +81.85%）；GBDT 預測器 AUC 0.917–0.922、ECE 0.003–0.004；MI300X 四個 baseline 端到端差距只有 1.6%；MI300X decode 佔 78–82%，端到端 headroom 3.76–4.41%。
- **F11** query 側訊號（注意力重要度）對跨請求重用 AUC 0.480–0.500（7 個模型）；κ_ssd 在兩平台都 < 1（3090：0.82–0.94；MI300X：0.72，即這些位置重算比 SSD 取回便宜）；SSD warm 成本實測 spread 90.1%；bootstrap 下「最佳 baseline」有 27% 會從 `tier_fs` 翻成 `cpu_lru`。

---

## §1 一覽表（先看這張）

「決策」欄：① 幾個 bit（AdaptCache 的戰場）② 放哪一層（Strata 機制／Bidaw 逐出／MTDS）③ 要用時重算還是載回（Cake）；「聯合」＝同時牽動多個。

| ID | 工具 | 決策 | 解釋（✓）／威脅（✗）哪個事實 | 查到的 KV 先例 | 新穎性風險 |
|---|---|---|---|---|---|
| A1 | Little's Law | ②③ | ✓F9（per-op 截距）✓F2 | Strata §3.1 | 高 |
| A2 | Amdahl's law | 聯合 | ✓F5 ✓F10；✗M4 口徑 | 本次未找到 | 低（但不是貢獻） |
| A3 | Roofline（+κ_crit） | ①③ | ✗「κ 跨硬體」主張；✗F5（decode 也可能受精度影響） | bottlenecks2026 的 κ_crit 等 4 篇 | 高 |
| A4 | 五分鐘法則／10-byte rule | ②③ | 故事定律 | MatKV（ICDE'26）的 ten-day rule | 中 |
| B1 | Delayed hits（+後續 8 篇） | ②③ | ✗F6（Belady 次序非延遲最優） | Strata | 高 |
| C1–C3 | GreedyDual／GDS／GDSF | ②③ | ✓F3（value/byte） | RAGCache PGDSF、KVCache-in-the-Wild | 高 |
| C4 | Landlord | ②③ | ✓F3；接 ski rental | 本次未找到 | 中 |
| C5 | LHD（hit density） | ②③ | ✓F3；✓F4 | 本次未找到 | 中 |
| C6 | LRB（Belady boundary） | 預測器 | ✓F4 ✓F10 | 本次未找到 | 中 |
| C7–C8 | AdaptSize／Hyperbolic | ② | 旁支 | 本次未找到 | 低 |
| D1–D5 | Flashield／Kangaroo／Baleen／CacheLib／TinyLFU | ②（SSD 准入） | SSD 寫入量未被計入 | Strata 引 Flashield（只為傳輸大小）；HBF-KV（LRU-K 准入） | 低–中 |
| E1–E7 | ARC／2Q／LIRS／CLOCK-Pro／S3-FIFO／SIEVE／LRU-K | ② | baseline；✗「LRU≈OPT」威脅 | KVCache-in-the-Wild、Ganjihal 2026、ICML'26 WS | 高（當貢獻） |
| F1–F4 | DEMOTE／ULC／Karma／Gill（PROMOTE + OPT-UB/LB） | ② | ✓F3（降級成本面）✓F6 | Bidaw 用 inclusive | 中 |
| G1–G4 | 整合預取與快取（Cao 四規則、TIP、Albers、Harmony） | ③ | ✗F6（含預取時 Belady 非最優） | Cake／Strata 為同類機制 | 中 |
| H1–H6 | Belady／Mattson／FOO／weighted-caching min-cost flow／NP-hardness／Belatedly | Oracle | ✓✗F6、F8 的 headroom 可信度 | Bidaw、Ganjihal 2026（hit-rate 形式） | 中 |
| I1–I7 | Che／TTL／階層分析／效用-Lagrange／MRC 工具／Talus／working set | 理論模型 | ✓F8（替代）✓F3 | Bidaw 的 weighted reuse distance；其餘本次未找到 | 中 |
| J1–J6 | ski rental／多態電源管理／elastic caching／competitive paging／multilevel | ②③(+①) | ✓F11（被支配的階）；線上保證 | InferCept／Continuum 為特例；RLT（ICLR'26） | 中 |
| K1–K11 | learning-augmented caching | 預測器 | ✓F4 ✓F10 ✓F11 | LCR/LARU、RLT | 中 |
| L1–L8 | Checkmate／DTR／Capuchin／SwapAdvisor／vDNN／POET／√n checkpoint + KV 類比 | ③ | ✓F1（重算鏈成本） | vLLM §7.3、InferCept、CacheOPT、HCache、Cake、KVPR、XQuant | 高（③ 已擁擠） |
| M1–M5 | 壓縮快取（Douglis、Wilson、zswap、Google SDFM、Meta TMO） | ① | ✓✓F3（1999 年就寫了同一件事）✓F8 | AdaptCache、EvicPress（KV 版） | 高 |
| N1–N5 | 物化視圖、WATCHMAN、Recycler、Nectar、MCKP／樹背包 | 聯合 | ✓F3（LP-dominance） | AdaptCache／EvicPress 用 MCKP | 中 |
| O1–O4 | M/G/1、SRPT、PASTA、LLM 排隊論文 | 排程 | ✓F11（SSD 變異的排隊懲罰） | 多篇 LLM 排程理論 | 低（對放置） |

---

## §2 逐條查證

### A 組：一條定律能講完的關係

#### A1. Little's Law
- **(a)** 穩態系統內平均個數 L ＝ 到達率 λ × 平均停留時間 W，與分佈無關。
- **(b)** J. D. C. Little, "A Proof for the Queuing Formula: L = λW," *Operations Research* 9(3):383–387, 1961. <https://doi.org/10.1287/opre.9.3.383>；50 週年回顧：Little, *Operations Research* 2011, <https://pubsonline.informs.org/doi/10.1287/opre.1110.0940>（摘要核對）。
- **(c)** ②③。Strata 已把它用在 I/O 併發（F7）。F9 正是它的實例：有效頻寬 `X = C·S/L`，而 `L = t0 + S/B`（t0 為每筆描述符的固定成本），C=1 時 `X = S/(t0 + S/B)`——小 S 時 X 由 t0 決定（見 §3.2 的算術驗算）。Strata 沒用的兩個用法（我們的判讀）：(i) 把「位元組」當顧客：GPU 平均駐留時間 `W_GPU = C_GPU / λ_B`（λ_B＝寫入 GPU 的位元組速率）——這就是 Che 近似裡 characteristic time 的 Little 形式；INT4 讓同一 C_GPU 裝 3.77× 的 block（F8），W 也跟著拉長；(ii) F2 的 65% 軟體成本若是 per-operation 的，Little 直接給出「要嘛提高併發 C、要嘛放大 S」兩條路（與 Strata page-first 同構）。
- **(d)** 有：Strata（OSDI'26 <https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang>）§3.1 原文「C = λ·L … X = C·S/L」（本地 `strata2026.txt` L295–305 全文核對）。本地其他 34 篇未見；web 搜尋「Little's law KV cache」未見其他 KV 論文。
- **(e)** 模擬器一致性檢查：每個時間窗記錄 GPU 常駐位元組 B(t)、進入 GPU 的位元組速率 λ_B(t)、每個 block 的實際駐留時間 W；驗 `mean(B) ≈ mean(λ_B)·mean(W)`（Little 對任何策略都成立，不成立代表模擬器記帳有誤）。順便得到 W_GPU，供 §4 定律二用。F9 的部分：用微基準的 (S, X) 表擬合 `S/X = t0/C + S/(C·B)`，分別估 t0 與 C（兩個未知數，只看吞吐分不開，要同時量每筆延遲）。
- **(f)** 高（I/O 併發／傳輸大小＝Strata §3.1 已做；vLLM SOSP'23 §7.3 也寫過小 block 造成大量小傳輸、壓低有效 PCIe 頻寬，見 L 組）。只能當工具，不能當貢獻；F9 的新意在「ROCm 上 Triton 快速路徑被停用」這個具體實作事實與它對 κ 的影響，不在定律本身。

#### A2. Amdahl's law
- **(a)** 只能把佔比 f 的部分加速 s 倍時，總加速 ≤ 1/((1−f)+f/s) ≤ 1/(1−f)。
- **(b)** G. M. Amdahl, "Validity of the single processor approach to achieving large scale computing capabilities," AFIPS SJCC 1967, pp. 483–485. <https://doi.org/10.1145/1465482.1465560>（摘要核對）。
- **(c)** 聯合。直接對應 F5：放置只影響 prefill → 端到端改善上限＝prefill 佔比。見 §4 定律一（含 M4 門檻換算）。F10 的「MI300X 端到端 headroom 3.76–4.41%」就是它在資料上的實例（§4 有一致性算術）。威脅：M4 的 15%／5% 門檻若沒指明是 prefill 還是端到端口徑，兩種口徑的門檻差 1/(1−f_d) 倍，即約 2–5.6 倍（由 F5、F10 的 decode 佔比算術得出）。
- **(d)** 本次未找到：本地 35 篇 grep「Amdahl」0 筆；web 搜尋「Amdahl + KV cache offloading」未見以它推導放置上限的論文。
- **(e)** 對每個 Oracle／policy run 同時輸出三個數：prefill 時間減少比 r_pf、decode 時間變化、端到端減少比 r_e2e；檢驗 `r_e2e ≈ (1−f_d)·r_pf`。若不成立，代表 decode 也被放置或精度影響（例如 GPU 上 INT4 讓 decode attention 讀的位元組變少，或 continuous batching 下 prefill 負載改變 decode 的 TBT），這本身就是要報告的發現。
- **(f)** 低（審稿人一定接受，但它不是貢獻）。

#### A3. Roofline（以及 κ_crit）
- **(a)** 可達效能 = min(峰值算力, 記憶體頻寬 × 運算強度)；ridge point = 峰值算力／頻寬。
- **(b)** S. Williams, A. Waterman, D. Patterson, "Roofline: an insightful visual performance model for multicore architectures," *CACM* 52(4):65–76, 2009. <https://doi.org/10.1145/1498765.1498785>（摘要核對）。
- **(c)** ①③。**威脅（重要）**：W. Meng, B. Lee, H. Wang（arXiv 2601.19910 <https://arxiv.org/abs/2601.19910>，MLSys 投稿格式）以 roofline 推出 `κ_crit = (F_pf/B_kv) × (BW_PCIe/C_eff)`，明確拆成「模型因子 κ_M × 硬體因子 κ_HW」【原文，本地 `bottlenecks2026.txt` Eq.(5)(6) 全文核對】。若 Tiara 的 κ 定義成「每 token 重算時間 ÷ 每 token 傳輸時間」，它與 κ_crit 在形式上等價（我們的判讀）。Tiara 能守的差異只剩：(i) F1 的位置項 α·P（κ_crit 假設每 token FLOPs 固定）；(ii) 分層的有效頻寬（F2 的 65% 軟體、F9 的描述符固定成本——這兩條其實在說 κ 的「硬體因子」大半是軟體因子）；(iii) 精度改變 B_kv；若傳輸時間與位元組成正比，κ 約乘上 2（FP8）或 3.77（INT4 的容量倍數）——但 F9 顯示小描述符時傳輸時間並不與位元組成正比，這個倍數要實測。另一個威脅是對 F5：decode attention 是 memory-bound，若 GPU 上以 INT4 存 KV 且 kernel 直接讀低精度，decode 也可能變快（未量測）。
- **(d)** 有：本地 bottlenecks2026（arXiv 2601.19910）、ForesightKV（arXiv 2602.03203 <https://arxiv.org/abs/2602.03203>）、KVDrive（arXiv 2605.18071）、KV survey（arXiv 2607.08057 <https://arxiv.org/abs/2607.08057>）都用 roofline；bottlenecks2026 定義 κ_crit。
- **(e)** 不需新量測：把 M1／M2 已有的 (model, tier) 成本點畫在 κ_crit 的封閉式上，算「κ_crit 能解釋 Tiara 量到的 κ 變異的幾成」；殘差就是位置項與軟體項的貢獻——這是對審稿人最直接的回答。
- **(f)** 高（「模型 × 硬體」的封閉式已被 κ_crit 佔走；且 F9 顯示傳輸項變異主要來自軟體描述符粒度）。

#### A4. 五分鐘法則與 10-byte rule
- **(a)** 把一頁留在記憶體的「租金」等於每次從磁碟讀回的成本時，其重新參照間隔即損益平衡點：`BreakEvenReferenceInterval = (PagesPerMBofRAM / AccessesPerSecondPerDisk) × (PricePerDiskDrive / PricePerMBofDRAM)`【原文，Gray–Graefe 1997 Eq.(1)，全文核對】；同一篇原始論文的 **10-byte rule** 是「用記憶體換 CPU 時間」＝存 vs 重算的原型【原文：Gray–Graefe 1997 回顧 1987 原文「10-byte rule for trading CPU instructions off against DRAM」】。
- **(b)** J. Gray, G. F. Putzolu, "The 5 minute rule for trading memory for disc accesses and the 10 byte rule for trading memory for CPU time," SIGMOD 1987（*SIGMOD Record* 16(3):395–398）<https://doi.org/10.1145/38714.38755>；J. Gray, G. Graefe, *SIGMOD Record* 26(4):63–68, 1997 <https://doi.org/10.1145/271074.271094>（arXiv cs/9809005，全文核對）；G. Graefe, DaMoN 2007 <https://doi.org/10.1145/1363189.1363198>（ACM Queue 2008 版 <https://doi.org/10.1145/1413254.1413264>）；R. Appuswamy, R. Borovica-Gajic, G. Graefe, A. Ailamaki, ADMS@VLDB 2017 <https://renata.borovica-gajic.com/data/adms2017_5minuterule.pdf>；T. Zhang et al.（含 W.-m. Hwu），arXiv 2511.03944（2025；v1 標題 "From Minutes to Seconds…"）<https://arxiv.org/abs/2511.03944>【摘要：GPU／高效能 SSD 平台上 DRAM↔flash 門檻由分鐘崩到數秒】。
- **(c)** 聯合（②③，① 當作租金減半／÷3.77）。最適合 Tiara 講故事的單一句子：「KV 的五分鐘法則 T* 由 κ 決定，而 κ 會跨硬體與實作移動」（§4 定律二）。F9 讓這句更精確：分母的「存取成本」有 per-descriptor 截距，所以 T* 也隨描述符粒度移動（我們的判讀）。
- **(d)** 有：MatKV（K.-W. Shin, J. H. Park, M. Oh, Y. Jo, J. Do, S.-W. Lee, arXiv 2512.22195 <https://arxiv.org/abs/2512.22195>；arXiv 頁標註 ICDE 2026）以五分鐘法則推出 SSD 物化 KV vs GPU 重算的「ten-day rule」【原文，HTML 全文核對：只比 SSD 與重算兩個選項，明言排除 CPU DRAM 與 HBM，以美元成本計】。
- **(e)** 在 Mooncake trace 上對每個 block 算實際 reuse interval 分佈；以 F1、F2、M2 常數與待定的 GPU 影子價格 λ 算 T*(P)；畫「reuse interval CDF vs T*(P)」→ 直接讀出「最佳情況下多少比例的 reuse 值得留在 GPU」。（⚠️ CLAUDE.md 規則 6：Mooncake `hash_ids` 是 512-token block，位置 P＝index×512；載入函式要用請求長度交叉驗算。）
- **(f)** 中（MatKV 已把五分鐘法則帶進 KV；Tiara 要強調多層、位置相依、以時間而非美元計、且 T* 會移動）。

---

### B 組：Delayed hits

#### B1. Caching with Delayed Hits 及後續
- **(a)** 高吞吐時，同一物件在前一次 miss 尚未補回前又被請求，這些請求既非 hit 也非完整 miss；hit-rate 最優（Belady）不等於延遲最優。
- **(b)** N. Atre, J. Sherry, W. Wang, D. S. Berger, "Caching with Delayed Hits," SIGCOMM 2020. <https://doi.org/10.1145/3387514.3405883>【原文，全文核對：離線延遲最優演算法 Belatedly 以 min-cost multi-commodity flow（MCMCF）求解，比 Belady 最多降 45% 平均延遲；線上 MAD 的排序函數 `Rank(x) = AggDelay(x)/TTNA(x)`，並以 LRU／ARC／LHD 當 TTNA 估計器】。後續（皆摘要核對）：
  - P. Manohar, J. Williams, "Lower Bounds for Caching with Delayed Hits," arXiv 2006.00376 <https://arxiv.org/abs/2006.00376>（決定性線上演算法 competitive ratio 下界 Ω(kZ)，Z＝取回延遲／請求間隔）
  - C. Zhang, H. Tan, G. Li, Z. Han, S. H.-C. Jiang, X. Li, "Online File Caching in Latency-Sensitive Systems with Delayed Hits and Bypassing," INFOCOM 2022 <https://ieeexplore.ieee.org/document/9796969/>（CaLa，O(Z^{3/2} log K)-competitive）
  - B. Carleton, "Delayed Hits in Multi-Level Caches," SOSP 2021 poster <https://par.nsf.gov/biblio/10312968>
  - K. Gurushankar, N. G. Singer, B. Subercaseaux, "Latency Guarantees for Caching with Delayed Hits," INFOCOM 2025, arXiv 2501.16535 <https://arxiv.org/abs/2501.16535>（LRU 為 O(Zk)-competitive，首個緊的保證）
  - **learning 版本**：B. Jiang, Y. Yang, B. Jiang, "Optimizing latency for caching with delayed hits in non-stationary environment," *Performance Evaluation* 2025 <https://www.sciencedirect.com/science/article/abs/pii/S0166531625000227>（以 RNN+MLP 學非穩態到達過程；PDF 首頁核對）
  - N. Keren, G. Einziger, G. Scalosub, NSDI 2026 <https://www.usenix.org/conference/nsdi26/presentation/keren>；VA-CDH（變異感知）arXiv 2504.20335 <https://arxiv.org/abs/2504.20335>；隨機 miss 延遲，arXiv 2505.15531 <https://arxiv.org/abs/2505.15531>
- **(c)** ②③。DROP＋重算與 SSD 取回都很慢（F1、F2），同一 prefix 被多請求同時命中時就會出現 delayed hit；Oracle 若以 Belady 次序＋hit 計數建，不是時間最優 → 威脅 F6、以及 F8 的 headroom 數字（我們的判讀）。MAD 的 AggDelay/TTNA 就是「miss 成本 ÷ 距下次使用時間」的價值密度，與 F3 的「剩餘價值÷位元組」同一家族（我們的判讀）。F11 的 SSD 成本 90.1% spread 也對應 VA-CDH 的「變異」考量。
- **(d)** 有：Strata（OSDI'26）引用 Atre et al. 並處理 delay hit 與 bundle hit（本地 `strata2026.txt` L176、L735、L817 全文核對）。
- **(e)** 模擬器事件佇列加「in-flight 表」：某 block 的 SSD 讀取或重算尚未完成時，後到請求等待剩餘時間，而不是再發一次；開關此邏輯比較 LRU／value-per-byte／Oracle 的總時間 → 量化「忽略 delayed hit 讓 Oracle 高估或低估多少」。只改計時，不改策略。
- **(f)** 高（Strata 已帶入 KV；Tiara 只能沿用）。

---

### C 組：成本／大小感知快取

#### C1. GreedyDual
- **(a)** 每個物件帶 credit H＝其 miss 成本；逐出 H 最小者並把全體 H 減去該值（以全域 L 實作）；成本一致時退化成 LRU。
- **(b)** N. E. Young, "The k-server dual and loose competitiveness for paging," *Algorithmica* 11:525–541, 1994（SODA'91 初版）<https://doi.org/10.1007/BF01189992>【原文（摘要核對）：LRU 與 weighted caching 的 balance 演算法都是 LP primal-dual 演算法，GreedyDual 兩者皆推廣並對 weighted caching 有最優保證；容量 k 對上最優容量 h 時成本在 k/(k−h+1) 倍內】。
- **(c)** ③（每個 block 的 miss 成本 ＝ min(C_ssd, C_rc(P), C_cpu)，隨 P 與 κ 變）。解釋 F3：LRU 是 GreedyDual 在成本一致時的特例；Tiara 的成本依 F1 線性於 P、依 F11 隨平台改變，LRU 次序與成本無關是結構性缺陷（我們的判讀）。
- **(d)** 間接：RAGCache PGDSF、KVCache-in-the-Wild 的 GDFS 變體屬同家族（見 C3）。本次未找到直接用 Young 原版的 KV 論文。
- **(e)** 模擬器加 GreedyDual：H(b)=miss_cost(b)（F1／F2 常數），逐出後 L←H_evicted，O(log n) 優先佇列。若 GD 已接近 value/byte，F3 的改善應歸因於「成本感知」而非 Tiara 特有設計。
- **(f)** 高（只能當強 baseline）。

#### C2. GreedyDual-Size（GDS）
- **(a)** GreedyDual 加大小：H = L + cost/size。
- **(b)** P. Cao, S. Irani, "Cost-Aware WWW Proxy Caching Algorithms," USITS 1997. <https://www.usenix.org/conference/usits-97/cost-aware-www-proxy-caching-algorithms>
- **(c)** ①②③聯合。把 size 換成「在 GPU 上的實際佔用」（BF16＝s、INT4＝s/3.77），壓縮後 cost/size 變大 → 自然留在 GPU。這解釋 F3 為何改成「剩餘價值÷位元組」後 INT8 階從無效變有效（我們的判讀）。
- **(d)** 見 C3。
- **(e)** 同 C1，H＝L＋miss_cost／bytes_on_GPU(tier)，降精度時重算 H。
- **(f)** 高。

#### C3. GreedyDual-Size-Frequency（GDSF）
- **(a)** GDS 再乘存取頻率：H = L + Freq × Cost/Size。
- **(b)** L. Cherkasova, "Improving WWW Proxies Performance with Greedy-Dual-Size-Frequency Caching Policy," HP Labs Tech. Report HPL-98-69(R1), 1998. <https://www.researchgate.net/publication/228542715_Improving_WWW_proxies_performance_with_Greedy-Dual-Size-Frequency_caching_policy>（HP Labs 1998 報告索引：<http://shiftleft.com/mirrors/www.hpl.hp.com/techreports/98/index.html>）
- **(c)** 同 C2，加頻率。
- **(d)** **有（強）**：RAGCache（C. Jin, Z. Zhang, X. Jiang, F. Liu, X. Liu, X. Liu, X. Jin；arXiv 2404.12457 <https://arxiv.org/abs/2404.12457>；ACM TOCS <https://doi.org/10.1145/3768628>）的 PGDSF：`Priority = Clock + Frequency × Cost/Size`，Cost 是依「前序文件是否已快取」而變的重算時間（離線 profile 不同已快取／未快取長度的 prefill 時間），GPU 與 host 各有一個 clock【原文，全文核對 §4】。J. Wang et al., "KVCache Cache in the Wild," USENIX ATC'25（arXiv 2506.02634 <https://arxiv.org/abs/2506.02634>）以 GDFS 為起點改成 (ReuseProb, −Offset) 的字典序，並與 GDFS、LRU、LFU、FIFO、S3-FIFO 比較【原文，全文核對 §4.2；ReuseProb 由指數分佈擬合，Offset 使較後段 block 較早被逐出】。Marconi（R. Pan et al., MLSys'25, arXiv 2411.19379 <https://arxiv.org/abs/2411.19379>）在相關工作點名 GDSF（本地 `marconi2025.txt` L756）。
- **(e)** 直接把 PGDSF 放進 M3／M5 當 baseline，Cost 用 F1 的 `C_rc0 + α·P`——這是對 Tiara 最直接的「前人已做」檢驗。
- **(f)** 高（「位置相依重算成本 × 頻率 ÷ 大小」的 KV 版本已發表）。

#### C4. Landlord
- **(a)** 每個檔案有 credit，每步依大小對所有檔案收「租金」，credit 用完就逐出，被請求時補回；LRU 的推廣。
- **(b)** N. E. Young, "On-Line File Caching," *Algorithmica* 33(3):371–383, 2002（SODA'98）<https://doi.org/10.1007/s00453-001-0124-5>；arXiv cs/0205033【摘要核對：決定性線上演算法中有最優保證】。
- **(c)** ②③。「依大小與時間收租」＝GPU 位元組×時間的影子價格，是 §4 定律二的線上版本（我們的判讀）；精度階＝租金 ÷3.77。
- **(d)** 本次未找到（web 搜尋「KV cache Landlord」無 KV 結果；本地 0 筆）。
- **(e)** 每 block 每秒扣 λ·bytes、credit 初值＝miss 成本；λ 由 A1 的 W_GPU 或 I4 的 Lagrange 乘數給。與 C2 比較（理論上相近，但 Landlord 形式可直接讀成 ski-rental 的 timeout）。
- **(f)** 中。

#### C5. LHD（Least Hit Density）
- **(a)** 以 `hit density = 命中機率 ÷ (物件大小 × 預期剩餘駐留時間)` 排序，逐出最低者；以年齡的條件分佈估計。
- **(b)** N. Beckmann, H. Chen, A. Cidon, "LHD: Improving Cache Hit Rate by Maximizing Hit Density," NSDI 2018. <https://www.usenix.org/conference/nsdi18/presentation/beckmann>【原文，全文核對 Eq.(2)–(6)；刻意用「剩餘」壽命避免 sunk-cost】。
- **(c)** ②③。F3 的「剩餘價值÷位元組」少了「÷預期剩餘駐留時間」；LHD 的推導說明正確單位是「每位元組·時間的收益」（我們的判讀）。對 F4／F11：LHD 不靠 ML 預測器，只用「依類別（例如位置 P 分箱）的年齡分佈」——F11 說注意力幾乎完全由位置決定，正好說明「位置」應當是類別鍵而不是被學的特徵（我們的判讀）。
- **(d)** 本次未找到。
- **(e)** 以 P 分箱估每箱「年齡 a 下的 hit／eviction 年齡分佈」，實作 LHD×cost（把 hit 換成 saved cost），對照 value/byte 與 GBDT。
- **(f)** 中。

#### C6. LRB（Learning Relaxed Belady）
- **(a)** 用 ML 模仿「relaxed Belady」：只需逐出下一次請求超過 Belady boundary 的物件，不必找最遠者；用 good decision ratio 評估決策。
- **(b)** Z. Song, D. S. Berger, K. Li, W. Lloyd, NSDI 2020. <https://www.usenix.org/conference/nsdi20/presentation/song>【原文，全文核對：Belady boundary＝Belady MIN 所逐出物件的最小 time-to-next-request；good decision ratio＝被逐出物件的下一次請求超過該邊界的比例】。
- **(c)** 預測器。直接解釋 F4 與 F10：AUC 0.9995（3090）／0.917–0.922（MI300X）衡量的是「會不會再用」，逐出要的是「是否超過 Belady boundary」，兩者標籤不同（我們的判讀）。
- **(d)** 本次未找到（web 與本地皆無）。
- **(e)** 離線跑一次 Belady 得邊界；對現有 GBDT 在每次逐出事件算 good decision ratio；把標籤改成「next use > boundary」重訓，比較 AUC 與 good decision ratio 的落差。成本＝一次 Belady＋一次重訓。
- **(f)** 中。

#### C7. AdaptSize
- **(a)** 以 `e^{−size/c}` 機率准入，c 由 Markov 模型線上調。
- **(b)** D. S. Berger, R. Sitaraman, M. Harchol-Balter, NSDI 2017. <https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/berger>
- **(c)** ②（准入）。對等大小 block 用處小，但對「整條 prefix／整個請求是否寫入 SSD」有用（我們的判讀）。
- **(d)** 本次未找到。
- **(e)** SSD 階以請求長度為 size 做機率准入，量 SSD 寫入位元組與總時間的 trade-off。
- **(f)** 低。

#### C8. Hyperbolic caching
- **(a)** 優先度＝自進入快取以來的請求數 ÷ 在快取中的時間（p_i = n_i/t_i），可乘成本變 cost-aware（p′_i = c_i·p_i）。
- **(b)** A. Blankstein, S. Sen, M. J. Freedman, USENIX ATC 2017. <https://www.usenix.org/conference/atc17/technical-sessions/presentation/blankstein>【原文，全文核對 Eq.(1) 與 §3.1】。
- **(c)** ②③；與 MAD 的 AggDelay/TTNA 同形（我們的判讀）。
- **(d)** 本次未找到。
- **(e)** 一行：p = miss_cost·n/t，當便宜 baseline。
- **(f)** 低–中。

---

### D 組：Flash 快取的准入與寫入量／耐久度

> 共通點（我們的判讀）：KV block 一旦算出就**不可變**，所以 flash 快取文獻最麻煩的「更新造成重寫」在 KV 不存在；剩下的是「寫進去之後會不會被讀」的准入問題，以及 SSD 階 per-op 軟體成本（F2）。Tiara 目前的成本模型似乎沒有把 SSD 寫入量／耐久度當成本（未查證 Tiara 程式，只依 F1–F11 推測）。

#### D1. Flashield
- **(a)** 以 DRAM 當濾網，用輕量 ML 預測「會被多次讀且不會被更新」的物件，只把它們以大塊順序寫入 flash。
- **(b)** A. Eisenman, A. Cidon, E. Pergament, O. Haimovich, R. Stutsman, M. Alizadeh, S. Katti, NSDI 2019. <https://www.usenix.org/conference/nsdi19/presentation/eisenman>
- **(c)** ②（SSD 准入）；大塊順序寫同時攤提 per-op 軟體成本（F2、F9）。
- **(d)** Strata 引用 Flashield，但只用來支撐「SSD 需要大傳輸量」（本地 `strata2026.txt` L318–324）；本次未找到把 Flashield 准入觀念用於 KV 的論文。
- **(e)** 模擬器記錄每小時寫入 SSD 的位元組、以及「寫入後從未讀回」的比例；後者高就代表需要准入控制。對照 NVMe 規格書的 DWPD（本機型號規格未查證）。
- **(f)** 低–中。

#### D2. Kangaroo
- **(a)** 小物件 flash 快取：大型 set-associative（省 DRAM 索引）＋小型 log-structured（省 flash 寫）＋閾值准入。
- **(b)** S. McAllister, B. Berg, J. Tutuncu-Macias, J. Yang, S. Gunasekar, J. Lu, D. S. Berger, N. Beckmann, G. R. Ganger, SOSP 2021（Best Paper）。<https://www.microsoft.com/en-us/research/publication/kangaroo-caching-billions-of-tiny-objects-on-flash/>；程式 <https://github.com/saramcallister/Kangaroo>
- **(c)** ②；若 SSD 階以小 block 為 I/O 單位，索引與寫入放大都是問題；F2 的 lookup 成本即索引成本（我們的判讀）。
- **(d)** 本次未找到。
- **(e)** 統計 SSD 階每次 I/O 大小分佈與 metadata 查詢次數，看 F2 的 lookup／cascade／promotion 哪一項與 I/O 次數成正比。
- **(f)** 低。

#### D3. Baleen
- **(a)** flash 快取的 ML 准入＋預取，以「episodes」為訓練用駐留模型，直接最佳化端到端指標（Disk-head Time）而非 miss ratio。
- **(b)** D. L.-K. Wong, H. Wu, C. Molder, S. Gunasekar, J. Lu, S. Khandkar, A. Sharma, D. S. Berger, N. Beckmann, G. R. Ganger, FAST 2024. <https://www.usenix.org/conference/fast24/presentation/wong>
- **(c)** ②。對 F4／F10：訓練目標應是系統時間，不是 AUC；episode 提供「一段駐留內值不值得」的標籤定義（我們的判讀）。
- **(d)** 本次未找到。
- **(e)** 標籤改為「以 episode 為單位：留在 GPU 能省下的時間 > 佔用代價」重訓，以總時間評估。
- **(f)** 低–中。

#### D4. CacheLib
- **(a)** Meta 通用快取引擎，DRAM＋flash 混合；flash 准入預設用固定機率 p 控制寫入速率；物件已在 flash 且在 DRAM 期間未修改就不重寫。
- **(b)** B. Berg, D. S. Berger, S. McAllister, I. Grosof, S. Gunasekar, J. Lu, M. Uhlar, J. Carrig, N. Beckmann, M. Harchol-Balter, G. R. Ganger, OSDI 2020. <https://www.usenix.org/conference/osdi20/presentation/berg>【原文，全文核對：固定機率准入、未修改不重寫、flash 耐久度】。
- **(c)** ②。KV 不可變 → inclusive SSD 階（已在 SSD 就不再寫）可直接省掉降級寫入（我們的判讀）；Bidaw（FAST'26）即採 inclusive caching 以避免逐出時的大量寫入【本地 `bidaw2026.txt` L600–602】。
- **(d)** InfiniGen（W. Lee et al., OSDI'24, arXiv 2406.19707 <https://arxiv.org/abs/2406.19707>）參考文獻列有 CacheLib（用途未核對）；Bidaw 的 inclusive 作法同理。
- **(e)** 切換 SSD 階 inclusive／exclusive，量 SSD 寫入量與 CPU／SSD 容量需求。
- **(f)** 低。

#### D5. TinyLFU
- **(a)** 以近似 LFU（Bloom-filter 系頻率草圖）決定新物件是否值得取代受害者（admission）。
- **(b)** G. Einziger, R. Friedman, B. Manes, "TinyLFU: A Highly Efficient Cache Admission Policy," *ACM TOS* 13(4):35, 2017. <https://doi.org/10.1145/3149371>；arXiv 1512.00727
- **(c)** ②；GPU 階准入：一次性長 prompt 的 block 不該擠掉熱 prefix（我們的判讀）。
- **(d)** 本次未找到 KV 論文用 TinyLFU。相近：Peng et al., arXiv 2609.07175（生成式推薦的 HBF KV，以 admission-controlled LRU-K 解耦寫入與 miss，理由是 flash 耐久度）<https://arxiv.org/abs/2609.07175>（摘要核對）。
- **(e)** GPU 階前加 TinyLFU 准入（Count-Min sketch），量總時間。
- **(f)** 低。

---

### E 組：經典與新式替換策略（hit-ratio 導向、成本無感）

共同 (c)：這些策略都不看成本，無法解釋 F1／F2 的成本結構；它們的用處是 (i) baseline（Tiara M3 已有 lru／arc）；(ii) **威脅**：Ganjihal（2026）在公開 LLM trace 上發現 recency 家族已在 Belady 的 0.9 個百分點內（見 §7），代表 hit-ratio 層面沒有 headroom，Tiara 的 headroom 只能來自成本異質性；(iii) LIRS 指出的「loop 比快取稍大時，LRU 總是逐出最快要用的 block」——若單請求長上下文的 KV 大於 GPU、且 attention 要依序掃過所有前序 KV（FlexGen／InfiniGen 類的部分駐留設定），存取就是 loop，LRU 是最壞選擇（我們的判讀；Tiara 目前走 vLLM 的跨請求重用，未必適用）；(iv) S3-FIFO／SIEVE 的 quick demotion：多數物件只被存取一次，應盡早逐出——對 KV 的含義是「新寫入且沒被重用的 block 應先降級」，與 F3 的「降級最冷者」方向不同（我們的判讀）。
共同 (e)：用 libCacheSim 或自寫模擬器把全部跑一遍，但報**成本加權**指標而非 hit ratio。共同 (f)：當貢獻＝高；當 baseline＝必要。

| ID | (a) 一句話 | (b) 引用（皆摘要核對，另註） | (d) KV 先例 |
|---|---|---|---|
| E1 ARC | 線上、自我調整地平衡 recency 與 frequency，每請求常數成本 | N. Megiddo, D. S. Modha, FAST 2003 <https://www.usenix.org/conference/fast-03/arc-self-tuning-low-overhead-replacement-cache> | vLLM 內建 arc（Tiara M3 baseline）；Y. Shen et al., "Recency/Frequency Adaptive KV Caching," ICML 2026 AdaptFM workshop, arXiv 2606.21238 <https://arxiv.org/abs/2606.21238>（ARC 式分配） |
| E2 2Q | 以兩個佇列區分首次與重複存取；常數時間開銷、無需調參、表現與 LRU/2 相當 | T. Johnson, D. Shasha, VLDB 1994, pp. 439–450 <https://dblp.org/rec/conf/vldb/JohnsonS94.html> | 本次未找到 |
| E3 LIRS | 以 inter-reference recency 取代 recency | S. Jiang, X. Zhang, SIGMETRICS 2002 <https://dl.acm.org/doi/10.1145/511399.511340>【全文核對：loop 比快取稍大時 LRU 總是逐出最快要用的 block】 | 本次未找到 |
| E4 CLOCK-Pro | 把 LIRS 精神做成 CLOCK 成本 | S. Jiang, F. Chen, X. Zhang, USENIX ATC 2005 <https://www.usenix.org/conference/2005-usenix-annual-technical-conference/clock-pro-effective-improvement-clock-replacement> | 本次未找到 |
| E5 S3-FIFO | 三個靜態 FIFO；小 FIFO 過濾一次性物件（quick demotion） | J. Yang, Y. Zhang, Z. Qiu, Y. Yue, K. V. Rashmi, SOSP 2023 <https://doi.org/10.1145/3600006.3613147>【全文核對摘要：多數物件在短窗內只被存取一次，關鍵在盡早逐出】 | KVCache-in-the-Wild 以之為 baseline；Ganjihal 2026 比較 |
| E6 SIEVE | 單一 FIFO＋visited bit＋移動的 hand；lazy promotion + quick demotion | Y. Zhang, J. Yang, Y. Yue, Y. Vigfusson, K. V. Rashmi, NSDI 2024 <https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo>（全文核對 §2.3） | Ganjihal 2026 比較 |
| E7 LRU-K | 以最近 K 次存取時間估 inter-arrival | E. J. O'Neil, P. E. O'Neil, G. Weikum, SIGMOD 1993 <https://doi.org/10.1145/170035.170081> | HBF-KV（arXiv 2609.07175）以 LRU-K 做准入 |

補充（新式、學習參數的啟發式）：H. Xia, W. Nixon, B. D. Marthen, P. Bhandari, J. Yang, "Learning-Augmented Heuristics," OSDI 2026 <https://www.usenix.org/conference/osdi26/presentation/xia>（arXiv 2608.27975）：學的是啟發式的**快取層級參數**而非逐物件預測；S4-FIFO 最壞 trace 只比 FIFO 多 0.8% miss【摘要核對】——見 K 組，對 F10 很關鍵。

---

### F 組：多層 exclusive caching

#### F1. DEMOTE（"My cache or yours?"）
- **(a)** 上層逐出時把 block 「降級」送到下層，使兩層 exclusive，總有效容量＝兩層相加。
- **(b)** T. M. Wong, J. Wilkes, USENIX ATC 2002. <https://www.usenix.org/conference/2002-usenix-annual-technical-conference/my-cache-or-yours-making-storage-more-exclusive>
- **(c)** ②。GPU→CPU→SSD 的每次降級都是一次 PCIe／NVMe 傳輸；F3 的 780,108 次降級正是 DEMOTE 的成本面（我們的判讀）。
- **(d)** 本次未找到；Bidaw 反而採 inclusive。
- **(e)** 比較 inclusive vs exclusive（DEMOTE）在 CPU／SSD 階的寫入與命中。
- **(f)** 低。

#### F2. ULC
- **(a)** 由 client 依「階層局部性強度」（reuse distance）指揮各層放置與替換。
- **(b)** S. Jiang, X. Zhang, ICDCS 2004, pp. 168–177. <https://ieeexplore.ieee.org/document/1281581/>
- **(c)** ②。以 stack distance 分段對應 `C_GPU`、`C_GPU+C_CPU`… 直接決定放哪層（我們的判讀）。
- **(d)** 本次未找到。
- **(e)** 零參數理論 baseline：`d(b) ≤ C_g → GPU；≤ C_g+C_c → CPU；否則依 P* 選 SSD 或 DROP`（d 用 Mattson stack distance，I5）。
- **(f)** 低–中。

#### F3. Karma
- **(a)** 用應用 hint 對多層做全域、exclusive 的分配與替換。
- **(b)** G. Yadgar, M. Factor, A. Schuster, FAST 2007, pp. 169–184. <https://www.usenix.org/conference/fast-07/karma-know-it-all-replacement-multilevel-cache>（摘要：與 LRU、2Q、ARC、MultiQ、LRU-SP、Demote 比較）
- **(c)** ②。LLM 服務有強 hint（session、tool call、prompt 結構）。
- **(d)** Karma 本身未見；hint 型 KV：Continuum（依 tool call 設 TTL，J 組）、CacheWise（tool call metadata，arXiv 2606.16824 <https://arxiv.org/abs/2606.16824>）。
- **(e)** 以 Mooncake trace 的 session 結構當 hint 分區管理。
- **(f)** 低。

#### F4. Gill：PROMOTE 與多層離線最優的上下界
- **(a)** 以機率性過濾決定哪些頁往上「promote」，取代 DEMOTE；並提出多層快取離線最優的上界 OPT-UB 與下界 OPT-LB。
- **(b)** B. S. Gill, "On Multi-level Exclusive Caching: Offline Optimality and Why Promotions Are Better Than Demotions," FAST 2008. <https://www.usenix.org/conference/fast08/technical-sessions/presentation/gill>【原文，全文核對：OPT-UB 定義第 i 層 hits `h_i = hitOPT(σ, Σ_{j≤i} S_j) − hitOPT(σ, Σ_{j<i} S_j)`、不計降級成本，證明沒有策略能更好；OPT-LB 是「逐層串接 Belady」的可實現策略；實驗中兩者相距 2.18%（兩層）／2.83%（三層）】。
- **(c)** Oracle（F6）。直接給多層 Oracle 上下界的骨架；PROMOTE 的觀點（下層 inclusive、只在命中時往上）與 CacheLib／Bidaw 一致（我們的判讀）。
- **(d)** 本次未找到。
- **(e)** 在 Mooncake trace 上算 OPT-UB（Belady 於 C_g 與 C_g+C_c）與 OPT-LB（串接 Belady），hit 數乘各層成本得時間的上下界——對 uniform 成本才嚴格；位置相依成本見 H3／H4。
- **(f)** 中（KV 未見，但只是帶入既有方法）。

---

### G 組：整合式預取與快取

#### G1. Cao–Felten–Karlin–Li 的四規則
- **(a)** 最優的預取＋快取策略必須滿足四規則：Optimal Prefetching、Optimal Replacement、**Do No Harm**（不可為了預取 B 丟掉比 B 更早要用的 A）、First Opportunity；並給 aggressive 與 conservative 兩策略。
- **(b)** P. Cao, E. W. Felten, A. R. Karlin, K. Li, "A Study of Integrated Prefetching and Caching Strategies," SIGMETRICS 1995 <https://doi.org/10.1145/223587.223608>【原文，全文核對四規則；conservative 在最優 elapsed time 的 2 倍內】；同作者 *ACM TOCS* 14(4), 1996 實作版 <https://collaborate.princeton.edu/en/publications/implementation-and-performance-of-integrated-application-controll/>（摘要核對）。
- **(c)** ③。「載回」可以提前做（promotion＝預取），但受 Do No Harm 約束；Cake 的雙向載入、Strata 的排程隱藏載入都屬此類（我們的判讀）。
- **(d)** 本次未找到 KV 論文引用四規則；機制上 Cake、Strata、InfiniGen 同類。
- **(e)** 模擬器加「aggressive promotion」（有空頻寬就把下一個要用的 block 往上搬）與 conservative 兩版，用 Oracle 的未來資訊跑 → 量化「允許預取的 Oracle 還能多拿多少」。
- **(f)** 中。

#### G2. TIP（Informed Prefetching and Caching）
- **(a)** 以應用 hint 與執行期 cost-benefit，在「預取 hinted block／快取 hinted block／LRU 快取」三者間動態分配 buffer。
- **(b)** R. H. Patterson, G. A. Gibson, E. Ginting, D. Stodolsky, J. Zelenka, SOSP 1995, pp. 79–95. <https://doi.org/10.1145/224056.224064>
- **(c)** 聯合。TIP 的「每個 buffer 的邊際效益」與 F3 的價值密度、Wilson（M2）的 cost/benefit 同精神（我們的判讀）。
- **(d)** 本次未找到。
- **(e)** 以已知的 session 下一輪 prefix 當 hint，用 TIP 式邊際效益在「GPU 留存 vs 預取 buffer」間分配 GPU 記憶體。
- **(f)** 低–中。

#### G3. Albers–Garg–Leonardi
- **(a)** 整合預取／快取的最小 stall time 可用 LP／網路流求解。
- **(b)** S. Albers, N. Garg, S. Leonardi, "Minimizing stall time in single and parallel disk systems," *JACM* 47(6):969–986, 2000. <https://doi.org/10.1145/355541.355542>【摘要核對：D 個磁碟時，以至多 2(D−1) 個額外快取位置得到不超過最優的 stall time】。
- **(c)** Oracle。單一 I/O 通道（例如一條 PCIe）時額外位置為 0，含預取的離線最優是多項式可解的（我們的判讀，依摘要推得）。
- **(d)** 本次未找到。
- **(e)** 見 H3，把預取變數加進 LP。
- **(f)** 低。

#### G4. Harmony（Belady with prefetching）
- **(a)** 有預取器時，Belady MIN 最小化的是含預取的總 miss，而非 demand miss。
- **(b)** A. Jain, C. Lin, "Rethinking Belady's Algorithm to Accommodate Prefetching," ISCA 2018, pp. 110–123. <https://doi.org/10.1109/ISCA.2018.00020>
- **(c)** Oracle（F6）：Oracle 若允許提前載回，Belady 次序不是最小需求延遲的解。
- **(d)** 本次未找到。
- **(e)** 在 Oracle 中分開統計 demand miss 與 prefetch，檢查頻寬是否浪費在不需要的預取。
- **(f)** 低。

KV 端的同類機制（計算與載入重疊）：Cake（S. Jin, X. Liu, Q. Zhang, Z. M. Mao, arXiv 2410.03065 <https://arxiv.org/abs/2410.03065>，任務標示 ICML'25）雙向載入：從前面算、從後面載、兩邊會合（本地 `cake2025.txt` L428–481 全文核對）。

---

### H 組：離線最優、NP-hard 與上下界（對 F6、F8 最關鍵）

#### H1. Belady MIN
- **(a)** 逐出下一次使用最遠者；在等大小、等成本下 hit 數最優。
- **(b)** L. A. Belady, "A study of replacement algorithms for a virtual-storage computer," *IBM Systems Journal* 5(2):78–101, 1966. <https://doi.org/10.1147/sj.52.0078>
- **(c)** Oracle。對 Tiara **不是**最優：成本異質（F1）、精度階使大小可變（s、s/3.77）、有 delayed hit（B1）、可預取（G4）。
- **(d)** 有：Bidaw（FAST'26 <https://www.usenix.org/conference/fast26/presentation/hu-shipeng>）§3.3.2 以「給定未來 trace 的 Belady」模擬 hit-rate 上界【本地 `bidaw2026.txt` L845–870 全文核對】；Ganjihal 2026（§7）在 prefix-block 模擬器算 Belady 最優。
- **(e)** 在 Mooncake 上算 block 級 Belady hit ratio 上界，再算「Belady 排程的成本」vs 貪婪 Oracle 的成本——若貪婪 Oracle 成本低於 Belady 排程，證明成本異質性確實重要。
- **(f)** 工具。

#### H2. Mattson 的 stack algorithm 與 OPT stack
- 見 I5（stack distance／MRC）。*IBM Systems Journal* 9(2):78–117, 1970, <https://doi.org/10.1147/sj.92.0078>（DOI 經 doi.org 轉址至 IEEE Xplore 5388318 核對）。

#### H3. FOO／PFOO（min-cost flow 求可變大小最優的上下界）
- **(a)** OPT 在同一物件兩次請求之間不改變「是否快取」的決定 → 以區間 ILP 表示，再改寫為 min-cost flow；允許非整數決策得下界（FOO-L），忽略非整數決策得上界（FOO-U）；PFOO 用於大 trace。
- **(b)** D. S. Berger, N. Beckmann, M. Harchol-Balter, "Practical Bounds on Optimal Caching with Variable Object Sizes," *POMACS* 2(2), 2018（SIGMETRICS'18）<https://arxiv.org/abs/1711.03709>【原文，全文核對：outer edge 成本 1/s_i；在正式 trace 上顯示現行系統比 OPT 多 11–43% miss，而先前的離線界以為幾乎沒有改善空間】。
- **(c)** Oracle。把 F6「不保證最優」變成「最多差 x%」的標準方法。**對 F8 特別重要**：headroom＝(best baseline − Oracle)/best baseline；Oracle 是貪婪解，真正的 OPT 只會更低 → 貪婪 Oracle 量出的 headroom 是真 headroom 的**下界**。F8 的「INT4 後只剩 2.02%／3.52%」因此只證明「至少還有 2.02%」，不能證明「最多 2.02%」；要主張替代關係，需要 OPT 成本的**下界**（例如 FOO-L），把殘餘 headroom 夾住（我們的判讀）。前提：貪婪 Oracle 是同一成本模型下**可行**的排程；若它為了好算而放寬了某些約束（例如忽略頻寬爭用或容量），方向會反過來、可能高估 headroom——需先核對 Oracle 的實作（本文未讀程式）。
- **(d)** 本次未找到 KV 論文用 FOO（web「KV cache min-cost flow / offline optimal」只回到 FOO 本身）。
- **(e)** 取 Mooncake 若干時間窗，建 FOO 網路：每個請求一節點，inner edge 容量＝C_GPU，outer edge 成本改成 `miss_cost_i / bytes_i`，其中 miss_cost_i＝min(C_ssd, C_rc0+α·P_i, C_cpu)（我們的推導，非文獻：把 FOO 的單位成本推廣成異質成本）；用 networkx／OR-Tools min-cost flow 或 HiGHS LP 求下界；貪婪 Oracle 為上界；BF16 與 INT4（容量×3.77）各做一次 → 殘餘 headroom 的區間。注意：多層＋搬移成本時「區間內不改決定」未必成立（可在區間中途降級），此時要改用時間切段的 LP，仍是鬆弛（下界）但變數多。
- **(f)** 中（KV 未見；貢獻在建模：多層、位置相依成本、bit-width 維度）。

#### H4. 等大小、異質成本的離線 weighted caching＝min-cost flow（精確解）
- **(a)** 頁大小相同、fault 成本不同時，離線最優可歸約為 min-cost flow，多項式時間精確解。
- **(b)** M. Chrobak, H. Karloff, T. Payne, S. Vishwanathan, "New results on server problems," *SIAM J. Discrete Math.* 4(2):172–181, 1991（SODA'90）<https://doi.org/10.1137/0404017>（依搜尋摘要：以 min-cost flow 給 weighted paging 的多項式離線演算法；原文 PDF 未逐字核對）。
- **(c)** Oracle。**全 BF16 或全 INT4 時，Tiara 的 block 是等大小的**，只有成本異質（F1 的位置項）→ 單層 GPU 的離線最優可以精確求，不必貪婪（我們的判讀）。混合精度才回到 H5 的 NP-hard。前提：每個 block 的 miss 成本與其他 block 在不在場無關；prefix 鏈重算（L2）會打破此前提，但若每次 miss 以「最便宜的單 block 成本」計，結果仍是真實成本的**下界**。
- **(d)** 本次未找到。
- **(e)** 先做「單層 GPU＋每次 miss 以 min(C_cpu, C_ssd, C_rc(P)) 計、忽略 CPU 容量」的精確 OPT——這是多層問題的一個**下界**；若只用 min(C_ssd, C_rc(P))，則是「無 CPU 階」系統的精確解。量貪婪 Oracle 與它的差距 → 直接回答 F6。
- **(f)** 中。

#### H5. NP-hard 與近似
- **(a)** 可變大小的一般快取（general caching）離線最優是 strongly NP-hard，連 fault 模型與頁大小 {1,2,3} 都是。
- **(b)** M. Chrobak, G. J. Woeginger, K. Makino, H. Xu, "Caching Is Hard—Even in the Fault Model," *Algorithmica* 63:781–794, 2012 <https://doi.org/10.1007/s00453-011-9502-9>；L. Folwarczný, J. Sgall, "General Caching Is Hard: Even with Small Pages," *Algorithmica* 79(2):319–339, 2017 <https://doi.org/10.1007/s00453-016-0185-0>（arXiv 1506.07905）；近似：A. Bar-Noy, R. Bar-Yehuda, A. Freund, J. Naor, B. Schieber, *JACM* 48(5):1069–1090, 2001 <https://doi.org/10.1145/502102.502107>（常數倍近似，應用含 general caching）；S. Irani, *Algorithmica* 33(3):384–409, 2002 <https://doi.org/10.1007/s00453-001-0125-4>（兩種成本模型的離線 O(log k) 近似、隨機線上 O(log² k)）。
- **(c)** Oracle。混合精度（s 與 s/3.77 並存）＝可變大小 → 求最優是 NP-hard，所以貪婪 Oracle 合理，但一定要附界（H3）。另一個結構性陷阱：prefix 相依使 block 的價值互補（子節點要祖先在才有用），貪婪法常見的 (1−1/e) 保證靠的是次模性，這裡不成立（我們的判讀；參見 N1、N5 的樹背包）。
- **(d)** AdaptCache 原文承認「greedy and not necessarily optimal … NP-hard」（協調者 `PAPERS_EXPLAINED_20260920.md` L426 引）；EvicPress 原文：MCKP、NP-hard、以貪婪解（本地 `evicpress2025.txt` L762 全文核對）。
- **(e)** 見 H3、H4。
- **(f)** 工具。

#### H6. Belatedly（延遲最優離線解）
- 見 B1：以 MCMCF 求 delayed hit 下的延遲最優。對 Tiara 的意義：若 in-flight 合併重要（B1(e)），Oracle 的正確目標是延遲，不是 hit 數。

---

### I 組：解析快取模型（「先做理論模型」的核心）

#### I1. Che's approximation（characteristic time）
- **(a)** LRU 快取近似成「每個物件駐留固定時間 T_C 的 TTL 快取」，T_C 由 `Σ_i P(hit_i)·size_i = C` 的定點決定；快取像一個截止頻率為 1/T_C 的低通濾波器。
- **(b)** H. Che, Y. Tung, Z. Wang, "Hierarchical Web caching systems: modeling, design and experimental results," *IEEE JSAC* 20(7), 2002. <https://dl.acm.org/doi/10.1109/JSAC.2002.801752>（摘要核對：兩層 LRU、characteristic time、低通濾波的比喻）；理論正當化：C. Fricker, P. Robert, J. Roberts, ITC 2012, arXiv 1202.3974 <https://arxiv.org/abs/1202.3974>。
- **(c)** 理論模型。T_C 就是 A1 的 W_GPU；也是 I4 的 Lagrange 乘數的倒數形式（Dehghan et al.）。配合 KVCache-in-the-Wild 的「每類請求 reuse time 近似指數分佈」【原文】，可以寫出每層 hit ratio 的封閉式（我們的判讀）。
- **(d)** 本次未找到 KV 論文用 Che 近似（web「Che approximation / characteristic time + KV」無結果）。
- **(e)** 以 Mooncake trace 依位置或請求類別分組，擬合 reuse time 分佈；解 T_C 定點，預測 LRU 在 C 與 3.77C 的 hit ratio 與成本；對照模擬器的 LRU。⚠️ Che／IRM 假設獨立到達，Mooncake 的突發性可能讓近似失準——這本身要先檢查（見 O3 PASTA）。
- **(f)** 中。

#### I2. TTL 快取分析
- **(a)** 每個物件帶計時器、到期逐出；TTL 比 LRU、FIFO、RND 更一般，可分析 cache network。
- **(b)** N. C. Fofack, P. Nain, G. Neglia, D. Towsley, "Analysis of TTL-based Cache Networks," VALUETOOLS 2012 <https://eudl.eu/doi/10.4108/valuetools.2012.250250>；同作者 "Performance evaluation of hierarchical TTL-based cache networks," *Computer Networks* 65:212–231, 2014 <https://www.sciencedirect.com/science/article/abs/pii/S1389128614001108>；D. Berger, P. Gland, S. Singla, F. Ciucu, "Exact analysis of TTL cache networks," *Performance Evaluation* 79:2–23, 2014 <https://www.microsoft.com/en-us/research/publication/exact-analysis-of-ttl-cache-networks/>（皆摘要核對）。
- **(c)** 理論模型＋線上策略：多層 KV 可寫成「GPU 計時器 T_1 → CPU 計時器 T_2 → SSD／DROP」的 TTL 串聯，與 J2 的「在 lower envelope 斷點降級」完全同構（我們的判讀）。
- **(d)** 分析本身未見；TTL 機制用於 KV 的有 Continuum（J 組）。
- **(e)** 把 T_1、T_2 當兩個旋鈕在模擬器掃描，與 J2 算出的斷點比較。
- **(f)** 中。

#### I3. 階層與互連快取的統一分析
- **(a)** 推廣 Che 的解耦技巧到多種策略與互連快取，考慮時間局部性。
- **(b)** V. Martina, M. Garetto, E. Leonardi, INFOCOM 2014，arXiv 1307.6702 <https://arxiv.org/abs/1307.6702>；期刊版 M. Garetto, E. Leonardi, V. Martina, *ACM TOMPECS* 2016 <https://doi.org/10.1145/2896380>（摘要核對）。
- **(c)** 理論模型（GPU／CPU／SSD 串聯）。
- **(d)** 本次未找到。
- **(e)** 以此計算 GPU→CPU 串聯（LRU／LRU 或 LRU／FIFO）的各層 hit ratio，驗模擬器。
- **(f)** 中。

#### I4. 效用驅動快取（Lagrange 乘數 ↔ characteristic time）
- **(a)** 每個內容一個 hit-probability 的效用函數，最大化總效用；最適解是 TTL 快取；LRU／FIFO 可逆推成特定效用的最大化器；characteristic time 對應容量約束的 Lagrange 乘數。
- **(b)** M. Dehghan, L. Massoulié, D. Towsley, D. Menasché, Y. C. Tay, "A Utility Optimization Approach to Network Cache Design," INFOCOM 2016，arXiv 1601.06838 <https://arxiv.org/abs/1601.06838>；期刊 *IEEE/ACM ToN* 2019 <https://dl.acm.org/doi/10.1109/TNET.2019.2913677>【原文，全文核對：「characteristic time … relates to the Lagrange multiplier corresponding to the cache capacity constraint」；Lagrangian `L(h,α)=Σ U_i(h_i) − α(Σ h_i − B)`】。相關：D. Carra, G. Neglia, P. Michiardi, "Elastic Provisioning of Cloud Caches: a Cost-aware TTL Approach," arXiv 1802.04696 <https://arxiv.org/abs/1802.04696>（儲存成本＋miss 成本的 TTL 動態調整；venue 未查證）。
- **(c)** **這是 §4 定律二的數學骨架**：把 GPU 容量約束 Lagrange 鬆弛 → 每個 block 獨立的 rent-or-buy，租金＝λ（Lagrange 乘數）；品質約束 ε 也能用第二個乘數 μ 同法鬆弛（我們的推導，非文獻）。
- **(d)** 本次未找到。
- **(e)** 以二分搜 λ 使「預期 GPU 佔用＝C_GPU」，得到每個 block 的最佳 TTL 與層級 → 不跑完整模擬就能預測總成本（§6 步驟 3）。
- **(f)** 中。

#### I5. Stack distance／MRC 與其工具（**對 F3、F8 最有解釋力**）
- **(a)** LRU 是 stack algorithm：對所有容量 C 同時成立「命中 ⇔ stack distance < C」，一次掃描得到整條 miss ratio curve（MRC）；OPT 亦可用 stack 方式處理（Mattson 1970 的結果；本文未逐字核對原文）。
- **(b)** R. L. Mattson, J. Gecsei, D. R. Slutz, I. L. Traiger, "Evaluation techniques for storage hierarchies," *IBM Systems Journal* 9(2):78–117, 1970 <https://doi.org/10.1147/sj.92.0078>；近似工具：C. A. Waldspurger, N. Park, A. Garthwaite, I. Ahmad, "Efficient MRC Construction with SHARDS," FAST 2015 <https://www.usenix.org/conference/fast15/technical-sessions/presentation/waldspurger>；非 LRU 策略的 MRC：C. A. Waldspurger, T. Saemundsson, I. Ahmad, N. Park, "Cache Modeling and Optimization using Miniature Simulations," USENIX ATC 2017 <https://www.usenix.org/conference/atc17/technical-sessions/presentation/waldspurger>（摘要：可建 ARC、LIRS、OPT 的縮小模擬 MRC）；X. Hu, X. Wang, L. Zhou, Y. Luo, C. Ding, Z. Wang, "Kinetic Modeling of Data Eviction in Cache"（AET），USENIX ATC 2016 <https://www.usenix.org/conference/atc16/technical-sessions/presentation/hu>。
- **(c)** ①②。**解釋 F3**：LRU 次序下「降級最冷者、下一次就被搬走」，壓縮段只是過渡狀態，壓縮住的位元組 U_c 約只有一個 block，等效容量幾乎沒變，增益 ≈ MRC 在 C 附近一小段的落差 ≈ 0；「價值÷位元組」讓壓縮後的 block 價值密度升高而留下，U_c 變大，才真正沿 MRC 往右移（§4 定律三）。**預測 F8**：INT4 把等效容量乘 3.77，若 3.77C 已接近 footprint（MRC 的平坦區），LRU 與 OPT 的 miss 都趨近 compulsory miss 下限，兩者間隙關閉 → 放置策略無事可做（§3.1）。
- **(d)** Bidaw（FAST'26）的「weighted reuse distance」（其他使用者 KV 的總大小）即位元組加權 stack distance【本地 `bidaw2026.txt` L438–441、L794 全文核對】。本次未找到「bit-width × MRC」或「KV 壓縮階的 MRC 分析」。
- **(e)** §6 步驟 2：Fenwick tree O(N log N) 算 block 級 LRU stack distance → MRC_LRU；對一組容量跑 Belady 得 MRC_OPT；再做成本加權版（每次 miss 乘 min(C_ssd, C_rc(P))）。標出 C_BF16、3.77·C_BF16、footprint 三條垂直線。
- **(f)** 中（原理是教科書；新意在把 bit-width 當成 MRC 上的位移並在 KV 上量化）。

#### I6. Talus／UCP／Cliffhanger（懸崖與凸包、邊際效用分配）
- **(a)** Talus：把單一存取流切成兩個分區，讓 miss 曲線對容量呈凸（移除懸崖）；UCP：依各應用 MRC 的邊際效用分配共享快取；Cliffhanger：以增量爬山沿 hit-rate 曲線梯度調分配。
- **(b)** N. Beckmann, D. Sanchez, HPCA 2015, pp. 64–75 <https://ieeexplore.ieee.org/document/7056022/>；M. K. Qureshi, Y. N. Patt, MICRO 2006, pp. 423–432 <https://dl.acm.org/doi/10.1109/MICRO.2006.49>；A. Cidon, A. Eisenman, M. Alizadeh, S. Katti, NSDI 2016 <https://www.usenix.org/conference/nsdi16/technical-sessions/presentation/cidon>（皆摘要核對）。
- **(c)** ①：「GPU 記憶體分多少給 BF16、多少給 INT4」就是 UCP 式的邊際效用分配問題；F8 的品質預算 q 掃描（remote S2 設計）等於在 MRC 上決定壓縮段大小（我們的判讀）。
- **(d)** 本次未找到。
- **(e)** 用 I5 的 MRC 直接算 q 的邊際效用曲線，預測 S2 掃描的形狀（§3.1）。
- **(f)** 低–中。

#### I7. Working set（Denning）
- **(a)** 程式在最近 τ 時間內參照的頁集合；以 τ 視窗而非容量來描述局部性。
- **(b)** P. J. Denning, "The working set model for program behavior," *CACM* 11(5):323–333, 1968. <https://doi.org/10.1145/363095.363141>
- **(c)** 理論模型：working set 的 τ 與 TTL／T_C 同義；footprint（見 I5）即 τ→∞ 的 working set。
- **(d)** 本地 HetMem（arXiv 2508.13231 <https://arxiv.org/abs/2508.13231>）、KVDrive、InfiniGen 以「working set」一詞描述 KV（用法未逐一核對）。
- **(e)** 算 Mooncake 的 W(τ) 曲線，與 C_BF16、3.77C 比較。
- **(f)** 低。

---

### J 組：線上演算法（rent-or-buy 家族最關鍵）

#### J1. Ski rental（rent-or-buy）
- **(a)** 不知道要用多久時，「租」到累計租金等於「買」價再買，最壞只付最優的 2 倍；隨機化可到 e/(e−1)。
- **(b)** A. R. Karlin, M. S. Manasse, L. Rudolph, D. D. Sleator, "Competitive snoopy caching," *Algorithmica* 3:79–119, 1988（FOCS'86）<https://doi.org/10.1007/BF01762111>；A. R. Karlin, M. S. Manasse, L. A. McGeoch, S. Owicki, "Competitive randomized algorithms for nonuniform problems," *Algorithmica* 11:542–571, 1994 <https://doi.org/10.1007/BF01189993>【摘要核對：spin-block 等問題隨機 e/(e−1)≈1.58 最優，決定性最佳為 2】。
- **(c)** ②③。「留在 GPU（租）vs 丟掉之後重算或從 SSD 取回（買）」正是 rent-or-buy；spin-block（等 vs 阻塞）與「tool call 期間留不留 KV」同構（我們的判讀）。
- **(d)** 本次未找到以 ski rental 命名／分析的 KV 論文（web 與本地皆 0 筆）。結構等價的特例：InferCept（L 組）以 Preserve／Discard／Swap 的「GPU 記憶體浪費（GB·時間）」選最小者；Continuum（H. Li, R. He, Q. Mang et al., arXiv 2511.02230 <https://arxiv.org/abs/2511.02230>）以「重載成本＋排隊延遲」對「TTL 期間佔用 GPU 的代價」決定 TTL（`refs_txt/continuum.txt` 全文核對）——兩者都是「已知或預測區間長度」的 rent-or-buy，沒有 competitive 分析（我們的判讀）。
- **(e)** 對每個 block 的每段閒置期 Δ（trace 可得）算離線最優 min(租到底, 立即買)，再算線上 break-even 策略的成本，驗證「≤2×」在實際 trace 上的真實比例（通常遠小於 2）。
- **(f)** 中。

#### J2. 多態電源管理的 lower envelope（**多層 KV 的正確抽象**）
- **(a)** 閒置期長度未知、有多個狀態（每狀態有耗能率 α_i 與喚醒成本 β_i）時，離線最優為 lower envelope `LE(t) = min_i {α_i·t + β_i}`；線上 LEA 在 LE 的斷點轉換狀態，決定性 2-competitive（且對所有多態裝置而言是緊的）。
- **(b)** S. Irani, S. Shukla, R. Gupta, "Online strategies for dynamic power management in systems with multiple power-saving states," *ACM TECS* 2(3):325–346, 2003 <https://doi.org/10.1145/860176.860180>【原文，全文核對 LE(t) 定義與 2-competitive；另有依學得的閒置分佈的機率型策略】；Z. Lotker, B. Patt-Shamir, D. Rawitz, "Rent, Lease or Buy: Randomized Algorithms for Multislope Ski Rental," STACS 2008 <https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.STACS.2008.1331>（期刊 *SIAM J. Discrete Math.* 26(2):718–736, 2012）；**learning-augmented 版**：A. Antoniadis, C. Coester, M. Eliáš, A. Polak, B. Simon, "Learning-Augmented Dynamic Power Management with Multiple States via New Ski Rental Bounds," NeurIPS 2021, arXiv 2110.13116 <https://arxiv.org/abs/2110.13116>（摘要核對）。
- **(c)** ②③（＋①）。狀態＝{GPU-BF16, GPU-INT4, CPU, SSD, DROP}，「耗能率」＝GPU 位元組×影子價格（BF16：λs；INT4：λs/3.77；CPU／SSD／DROP：對 GPU 為 0），「喚醒成本」＝要用時的存取成本（BF16：0；INT4：dequant＋品質懲罰；CPU／SSD：傳輸；DROP：C_rc0＋α·P）。**預測**：不在 lower envelope 上的狀態永遠不該被用——F11 的 κ_ssd<1 意味著在那些位置 SSD 的喚醒成本高於 DROP、而兩者租金都≈0，所以 SSD 狀態被 DROP 支配（直到 P>P*）（我們的判讀）。見 §4 定律二。
- **(d)** 本次未找到（「multi-state ski rental / lower envelope + KV」0 筆）。
- **(e)** §6 步驟 3：對每個 block 用實測常數畫 LE(Δ)，列出每個 P 的斷點序列；統計貪婪 Oracle 的實際決策是否落在斷點附近——若是，Oracle 的行為可被一條封閉式解釋。
- **(f)** 中（KV 未見此框架；但 InferCept／Continuum 已有部分結構，需明確區分：多層、位置相依、competitive 保證、學習增強版）。

#### J3. Linear Elastic Caching via Ski Rental（快取＋記憶體時間成本）
- **(a)** 總成本＝miss 成本＋記憶體足跡對時間的積分；與 ski rental 建立理論連結，對每頁給 TTL；分開最佳化逐出策略與 ski-rental 策略即足以最小化總成本；並提出輕量 ML ski-rental 演算法。
- **(b)** R. Kumar, T. Lipcon, M. Purohit, T. Sarlos, CIDR 2025. <https://vldb.org/cidrdb/papers/2025/p22-kumar.pdf>【原文，PDF 首頁全文核對】。
- **(c)** ②③。與定律二同構：GPU 記憶體雖不按時間計費，但其影子價格 λ 讓「足跡×時間」有成本（我們的判讀）。其「逐出與 ski rental 可分開最佳化」的結論，是把 Tiara 拆成「排序」與「門檻」兩件事的理論依據。
- **(d)** 非 KV（Google Spanner）；本次未找到 KV 應用。
- **(e)** 照其作法：固定 LRU 排序，只學「每個 block 的 TTL」（ski-rental 門檻），比較與 GBDT 全學的差異——這也是 F10 的一個低風險替代方案。
- **(f)** 中。

#### J4. Competitive paging／weighted／generalized caching
- **(a)** 線上快取相對最優的最壞比例：LRU 的 competitive 分析（Sleator–Tarjan）；weighted paging 的 GreedyDual（C1）與隨機 O(log k)；可變大小與成本的 generalized caching 隨機 O(log² k)→O(log k)。
- **(b)** D. D. Sleator, R. E. Tarjan, *CACM* 28(2):202–208, 1985 <https://doi.org/10.1145/2786.2793>；N. Bansal, N. Buchbinder, J. Naor, "A primal-dual randomized algorithm for weighted paging," *JACM* 59(4), 2012 <https://doi.org/10.1145/2339123.2339126>（O(log k)）；同作者 "Randomized competitive algorithms for generalized caching," STOC 2008 <https://doi.org/10.1145/1374376.1374412>（*SIAM J. Comput.* 41(2):391–414, 2012；O(log² k)）；A. Adamaszek, A. Czumaj, M. Englert, H. Räcke, SODA 2012, pp. 1681–1689 <https://dblp.org/rec/conf/soda/AdamaszekCER12.html>（O(log k)；TALG 版 <https://doi.org/10.1145/3280826>）（皆摘要核對）。
- **(c)** 線上保證：Tiara 若主張「在所有硬體上都不會比 X 差太多」，這組結果給出可以引用的最壞界形式（我們的判讀）。
- **(d)** 有（KV）：F. Wu, S. Silwal, Q. Zhang, "Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective," ICLR 2026 <https://openreview.net/forum?id=R7fv5NWfMm>（arXiv 2601.18999；隨機化 leaf-token 逐出 RLT，O(log n)-competitive；摘要核對）。
- **(e)** 理論性，無需實驗；可在論文 related work 定位。
- **(f)** 中（ICLR'26 已把 competitive 分析帶入 KV prefix 逐出）。

#### J5. Multilevel caching 的 competitive 分析
- **(a)** relaxed list update 作為「多個容量遞增、存取時間遞增的快取」的模型。
- **(b)** M. Chrobak, J. Noga, "Competitive algorithms for relaxed list update and multilevel caching," *J. Algorithms* 34(2):282–308, 2000. <https://www.sciencedirect.com/science/article/abs/pii/S0196677499910611>（摘要核對）。
- **(c)** ②：GPU／CPU／SSD 三層的最壞界理論。
- **(d)** 本次未找到。
- **(e)** 理論定位用。
- **(f)** 低。

#### J6. LLM 排程的 competitive／排隊結果（與 KV 記憶體約束）
- P. Jaillet, J. Jiang, K. Mellou, M. Molinaro, C. Podimata, Z. Zhou, "Online Scheduling for LLM Inference with KV Cache Constraints," arXiv 2502.07115 <https://arxiv.org/abs/2502.07115>（任意到達下決定性演算法無常數 competitive；給多項式時間演算法）；見 O 組其餘。

---

### K 組：學習增強演算法（algorithms with predictions）

#### K1. Lykouris & Vassilvitskii（predictive marker）
- **(a)** 以預測的下一次到達時間輔助 Marker；誤差小時比最壞下界好，且永遠被 O(log k) 封頂。
- **(b)** T. Lykouris, S. Vassilvitskii, ICML 2018 <http://proceedings.mlr.press/v80/lykouris18a/lykouris18a.pdf>；*JACM* 68(4):24, 2021 <https://doi.org/10.1145/3447579>（摘要核對，arXiv 1802.05399）。
- **(c)** 預測器：F4／F10 的「學了反而輸」＝沒有 robustness 的學習策略；此文示範如何把預測器包在有保證的演算法裡。
- **(d)** 見 K9。
- **(e)** 見 K4 的包裝實驗。
- **(f)** 中。

#### K2. Rohatgi
- D. Rohatgi, "Near-Optimal Bounds for Online Caching with Machine Learned Advice," SODA 2020, pp. 1834–1845. <https://www.semanticscholar.org/paper/Near-Optimal-Bounds-for-Online-Caching-with-Machine-Rohatgi/ae090c618017d4e9872e27fe6798b5d8ce4a94f2>（摘要核對）。(c)(e)(f) 同 K1。

#### K3. Wei（BlindOracle ⊕ LRU）
- **(a)** 把「盲目照預測做」的 BlindOracle 與最優 competitive 演算法黑盒組合；與 LRU 組合在決定性演算法中是最優的。
- **(b)** A. Wei, "Better and Simpler Learning-Augmented Online Caching," APPROX/RANDOM 2020 <https://doi.org/10.4230/LIPIcs.APPROX/RANDOM.2020.60>（arXiv 2005.13716；摘要核對：「combining BlindOracle with LRU is in fact optimal among deterministic algorithms」）。
- **(c)** 預測器：F10 的直接處方——不要讓 GBDT 單獨決策，而是「GBDT ⊕ tier_fs（或 LRU）」切換（我們的判讀）。
- **(d)** 見 K9。
- **(e)** 見 K4。
- **(f)** 中。

#### K4. Chłędowski et al.（實驗研究：切換即保險）
- **(a)** 實證：「盲目跟隨預測器或經典穩健演算法、誰變差就切換」的簡單方法，在預測器好時開銷很小、預測器失效時與經典方法競爭——便宜的最壞情況保險。
- **(b)** J. Chłędowski, A. Polak, B. Szabucki, K. Żołna, "Robust Learning-Augmented Caching: An Experimental Study," ICML 2021, pp. 1920–1930 <http://proceedings.mlr.press/v139/chledowski21a.html>（arXiv 2106.14693；摘要逐字核對）。
- **(c)** 預測器：F10（MI300X −9.7% ~ −47.3%）正是論文所說「預測器失效時」的情形；切換機制理論上會把輸的幅度壓到接近 tier_fs（我們的判讀，未實測）。
- **(d)** 見 K9。
- **(e)** 在模擬器實作「以滑動視窗比較 GBDT 策略與 tier_fs 的累積成本，誰低就跟誰」；在 3090 與 MI300X 兩組成本常數下各跑一次：期望 3090 保留大部分 +81.85%、MI300X 從 −9.7~−47.3% 拉回到接近 0（這是預測，不是結果）。
- **(f)** 中。

#### K5. Antoniadis et al.（untrusted predictions 的一般組合）
- A. Antoniadis, C. Coester, M. Eliáš, A. Polak, B. Simon, "Online metric algorithms with untrusted predictions," ICML 2020, pp. 345–355. <https://proceedings.mlr.press/v119/antoniadis20a.html>（摘要核對）。(c) 提供「在多個演算法間切換」的理論依據（metrical task systems 框架）。

#### K6. Weighted paging with predictions
- Z. Jiang, D. Panigrahi, K. Sun, ICALP 2020 <https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ICALP.2020.69>（*TALG* 18(4):39, 2022 <https://dl.acm.org/doi/10.1145/3548774>）【摘要：weighted paging 下，只給固定 lookahead 或只給每頁下一次請求都不足以突破下界，SPRP 預測模型可 2-competitive】；N. Bansal, C. Coester, R. Kumar, M. Purohit, E. Vee, "Learning-Augmented Weighted Paging," SODA 2022, pp. 67–89 <https://doi.org/10.1137/1.9781611977073.4>（arXiv 2011.09076；完美預測時 ℓ-competitive 決定性、O(log ℓ) 隨機，ℓ＝權重類別數）。
- **(c)** Tiara 的 miss 成本是異質的（weighted），**只預測「下一次何時用」在 weighted 情形理論上不夠**（Jiang et al. 原文結論）——這直接說明 F4 的「AUC 高但次序沒學到」不只是訓練問題，也是預測目標設計問題（我們的判讀）。
- **(f)** 中。

#### K7. Ski rental with predictions
- **(a)** 以信任參數 λ∈(0,1) 平衡：決定性演算法 (1+1/λ)-robust、(1+λ)-consistent；隨機版更好。
- **(b)** M. Purohit, Z. Svitkina, R. Kumar, "Improving Online Algorithms via ML Predictions," NeurIPS 2018 <https://proceedings.neurips.cc/paper/2018/hash/73a427badebe0e32caa2e1fc7530b7f3-Abstract.html>【原文，全文核對 Theorem 2.2】。
- **(c)** ②③：把 GBDT 的預測當成「閒置期長度」的預測，交給 J2 的 lower envelope 決策，並用 λ 控制信任度——可跨硬體的學習方式（我們的判讀）；多態版見 J2 的 NeurIPS'21。
- **(e)** 以 λ 掃描，在 3090 與 MI300X 常數下各畫 consistency–robustness 曲線。
- **(f)** 中。

#### K8. 從 OPT 學（Hawkeye、PARROT）與排序損失
- **(a)** Hawkeye 在過去存取上重建 Belady 決策當標籤；PARROT 以 imitation learning 模仿 Belady，並用 NDCG 近似的排序損失（以 reuse distance 為相關度）與 DAgger。
- **(b)** A. Jain, C. Lin, "Back to the Future: Leveraging Belady's Algorithm for Improved Cache Replacement," ISCA 2016 <https://doi.org/10.1109/ISCA.2016.17>；E. Z. Liu, M. Hashemi, K. Swersky, P. Ranganathan, J. Ahn, "An Imitation Learning Approach for Cache Replacement," ICML 2020, pp. 6237–6247 <http://proceedings.mlr.press/v119/liu20f/liu20f.pdf>【原文，全文核對 §4.3 ranking loss】；排序在 LLM 排程的對應：Y. Fu, S. Zhu, R. Su, A. Qiao, I. Stoica, H. Zhang, "Efficient LLM Scheduling by Learning to Rank," NeurIPS 2024 <https://proceedings.neurips.cc/paper_files/paper/2024/hash/6c8985579293e0209bdaa4f21bb1d237-Abstract-Conference.html>（摘要：精確長度難測但相對排序可測；以 Kendall's τ 度量與理想 SJF 的接近程度）。
- **(c)** 預測器：F4 的「AUC 問會不會用，逐出要誰先用」＝分類 vs 排序；PARROT／Fu et al. 指出應直接最佳化排序（NDCG／Kendall τ），且用 DAgger 處理「訓練分佈≠自己策略造成的分佈」（我們的判讀：這可能是 3090→MI300X 輸掉的另一原因，因為換硬體後策略造出的狀態分佈變了）。
- **(d)** 本次未找到 KV 論文用 PARROT 式排序損失於逐出（KVP，L. Moschella, L. Manduchi, O. Sener, arXiv 2602.10238 <https://arxiv.org/abs/2602.10238>，以 RL 目標排序 KV entry，屬 intra-request 壓縮）。
- **(e)** 對現有 GBDT 在每個逐出事件計算「候選集合內預測次序 vs 真實下次使用次序」的 Kendall τ 與 LRB 的 good decision ratio，取代 AUC 當主要指標。
- **(f)** 中。

#### K9. 已用於 LLM 的 learning-augmented LRU
- **(a)** LARU 以預測誤差偵測決定何時退回 LRU；系統 LCR；1-consistent、O(k)-robust。
- **(b)** P. Chen, J. Zhang, H. Zhao, Y. Zhang, S. Chen, J. Yu, X. Tang, Y. Wang, H. Li, J. Zou, G. Xiong, K. Chow, S. He, S. Deng, "Toward Robust and Efficient ML-Based GPU Caching for Modern Inference," arXiv 2509.20979 <https://arxiv.org/abs/2509.20979>（摘要核對：LLM 工作負載 P99 TTFT 降最多 28.3%）；同組 "Robustifying Learning-Augmented Caching Efficiently without Compromising 1-Consistency"（Guard），NeurIPS 2025，arXiv 2507.16242 <https://arxiv.org/abs/2507.16242>（robustness 2H_{k−1}+2、保持 1-consistency）。
- **(c)** 預測器：**這是 F10 的直接先例與威脅**——「學習式＋LRU 保底」在 LLM prefix caching 已有人做。
- **(d)** 有（上列）。
- **(f)** 中–高（Tiara 不能宣稱首個 robust learned KV caching）。

#### K10. Learning-Augmented Heuristics（學參數而非學逐物件預測）
- 見 E 組補充（OSDI'26）。**(c)**：F10 顯示「同程式碼、只換硬體就翻盤」；LAH 的作法是讓模型只學啟發式的少數快取層級參數（例如 tier_fs 的門檻、定律二的 λ），結構本身保證穩健（我們的判讀）。**(e)**：把 Tiara 的學習部分縮成「每平台學 λ 與 P* 修正量」兩個數，比較與 GBDT 全學的表現與跨平台可移植性。**(f)** 中。

#### K11. 「預測不划算」的直接反證
- S. Ganjihal, "When Prediction Doesn't Pay: Recency Is Near-Optimal for LLM Prefix Caches," 2026-08-27（alphaXiv 頁 <https://www.alphaxiv.org/abs/2608.llm-prefix-cache-recency-optimal>；**標準 arXiv 編號未能確認**）【摘要：OpenAssistant 與 CodeAct 兩個公開 trace、prefix-block 模擬器、Belady 最優 vs FIFO／LRU／LFU／SIEVE／S3-FIFO；recency 家族在實際容量壓力下距 OPT 0.9 個百分點內，LRU 達 OPT 的 99.4–100%；只看 hit ratio、**不含**異質 miss 成本】。
- **(c)** ✗F4／F10 的反面證據：在 hit-ratio 口徑下學習沒有空間 → Tiara 的學習只可能在「成本異質」（F1、F2）上贏；MI300X 上 baseline 間只差 1.6%（F10）與此一致（我們的判讀）。

---

### L 組：DNN 訓練記憶體的「換出 vs 重算」（與 DROP＋重算同構）與 KV 類比

#### L1. Checkmate
- **(a)** 把 rematerialization（哪些張量丟掉之後重算）寫成 MILP，以現成求解器求最優排程。
- **(b)** P. Jain, A. Jain, A. Nrusimha, A. Gholami, P. Abbeel, K. Keutzer, I. Stoica, J. E. Gonzalez, MLSys 2020 <https://proceedings.mlsys.org/paper_files/paper/2020/hash/0b816ae8f06f8dd3543dc3d9ef196cab-Abstract.html>（arXiv 1910.02653）。
- **(c)** Oracle：DROP 決策的精確 MILP 形式；每個 chunk 的重算成本用 F1 的線性式（我們的判讀）。
- **(d)** 本次未找到 KV 論文用 Checkmate 式 MILP；OrbitFlow（arXiv 2601.10729 <https://arxiv.org/abs/2601.10729>）以 ILP 決定保留哪些層的 KV（本地 `orbitflow2026.txt` L38、L135）。
- **(e)** 對單一長請求（「單請求 × context 遞增」壓力軸）建小型 MILP：變數＝每個 chunk 在每個時間點的狀態，求 prefill 最短時間 → 小規模的精確上界。
- **(f)** 中。

#### L2. DTR（Dynamic Tensor Rematerialization）
- **(a)** 線上貪婪 checkpointing，逐出 `h_DTR(t) = c(t) / [m(t) · s(t)]` 最小者（c＝投影重算成本、m＝記憶體、s＝staleness）；c(t) 含「evicted neighborhood」的重算鏈成本。
- **(b)** M. Kirisame, S. Lyubomirsky, A. Haan, J. Brennan, M. He, J. Roesch, T. Chen, Z. Tatlock, ICLR 2021 <https://ztatlock.net/pubs/2021-iclr-dtr/2021-iclr-dtr.pdf>（arXiv 2006.09616）【原文，全文核對：`c(t) = c0(t) + Σ_{t′∈e*(t)} c0(t′)`；Theorem 3.1：N 層線性網路、Ω(√N) 記憶體下 O(N) 運算；Theorem 3.2：任何決定性啟發式都存在網路需 Ω(N/B) 倍於最優靜態 checkpointing 的運算】。
- **(c)** ③＋F1／F3：h_DTR 正是「成本 ÷ (位元組 × 閒置時間)」的價值密度，與 LHD、F3 同家族；「evicted neighborhood」精準對應 M2 的 `recompute_chain.csv`：一個 block 的前綴也被丟掉時，重算成本是整條鏈（我們的判讀）。Theorem 3.2 提醒：純貪婪的線上 DROP 有最壞情形。
- **(d)** ArkVale（R. Chen et al., NeurIPS 2024 <https://papers.nips.cc/paper_files/paper/2024/hash/cd4b49379efac6e84186a3ffce108c37-Abstract-Conference.html>）參考文獻列有 DTR（用途未核對，本地 `arkvale2024.txt` L1108）。
- **(e)** 在模擬器實作 h_DTR（c 用 F1 常數＋前綴鏈成本、m＝GPU 位元組、s＝閒置時間）當 baseline，對照 value/byte。
- **(f)** 中–高（價值密度已被多方佔用；鏈成本的 KV 版可能仍是空的）。

#### L3–L7（其餘 swap/recompute 系統）
| ID | (a) | (b) | (c)(d)(f) |
|---|---|---|---|
| L3 Capuchin | 依執行期張量存取模式，逐張量選 swap 或 recompute | X. Peng, X. Shi, H. Dai, H. Jin et al., ASPLOS 2020 <https://doi.org/10.1145/3373376.3378505> | ③ 的逐物件二選一；InfiniGen 參考文獻列有它；風險中 |
| L4 SwapAdvisor | 聯合最佳化運算排程、記憶體配置、swap 決策 | C.-C. Huang, G. Jin, J. Li, ASPLOS 2020 <https://www.news.cs.nyu.edu/~jinyang/pub/swapadvisor-asplos20.pdf> | ②③ 聯合的先例；KV 未見；風險低 |
| L5 vDNN | 訓練時把特徵圖 offload 到 CPU、預取回 | M. Rhu, N. Gimelshein, J. Clemons, A. Zulfiqar, S. W. Keckler, MICRO 2016 <https://ieeexplore.ieee.org/iel7/7777315/7783693/07783721.pdf> | ② 的 GPU↔CPU 原型；風險低 |
| L6 POET | 把 rematerialization 與 paging 放進同一個 MILP（能耗最優、記憶體與時間約束） | S. G. Patil, P. Jain, P. Dutta, I. Stoica, J. E. Gonzalez, ICML 2022 <https://arxiv.org/abs/2207.07697> | **②③ 聯合 MILP 的最近先例**；Tiara 的六態動作空間＝POET 的「重算 or 換頁」＋精度（我們的判讀）；風險中 |
| L7 √n checkpointing | O(√n) 記憶體只多一次 forward | T. Chen, B. Xu, C. Zhang, C. Guestrin, arXiv 1604.06174 <https://arxiv.org/abs/1604.06174> | 理論參考；風險低 |

#### L8. KV 世界已有的「swap vs recompute」（③ 已擁擠——威脅）
- **vLLM／PagedAttention**（W. Kwon et al., SOSP 2023, arXiv 2309.06180 <https://arxiv.org/abs/2309.06180>）【原文，全文核對 §4.5、§7.3：swap 與 recompute 的效能「depend on the bandwidth between CPU RAM and GPU memory and the computation power of the GPU」；小 block 造成大量小傳輸、限制有效 PCIe 頻寬；recompute 開銷不隨 block 大小變】——**這是「κ 依硬體而變」在 LLM serving 中很早的定性陳述（是否為最早未查證），也是 F9 小描述符問題的早期觀察**。
- **InferCept**（R. Abhyankar, Z. He, V. Srivatsa, H. Zhang, Y. Zhang, ICML 2024 <https://proceedings.mlr.press/v235/abhyankar24a.html>）【原文，全文核對 Eq.(1)–(3)：WasteDiscard＝T_fwd(C)·(C+C_other)·M、WastePreserve＝T_INT·C·M、WasteSwap＝2·T_swap(C)·C_batch·M，取最小】——結構上就是已知區間長度的三態 rent-or-buy（我們的判讀）。
- **CacheOPT**（H. Shen, T. Sen, M. Tanaka, arXiv 2503.13773 <https://arxiv.org/abs/2503.13773>）【原文，全文核對：「For sequences exceeding the sweet spot length, swapping results in shorter times compared to recomputation」，以 profiling 找 sweet spot】——請求層級的 P* 類比，**對 F2 的 P* 構成新穎性壓力**。
- **HCache**（S. Gao, Y. Chen, J. Shu, EuroSys 2025 <https://doi.org/10.1145/3689031.3696072>，本地 `hcache2025.txt`）：從中間激活還原、平衡計算與 I/O。**Cake**（雙向）與 **KVPR**（C. Jiang, L. Gao, H. E. Zarch, M. Annavaram, arXiv 2411.17089 <https://arxiv.org/abs/2411.17089>，I/O-aware 部分重算）屬同類。**XQuant**（A. Tomar et al., arXiv 2508.10395 <https://arxiv.org/abs/2508.10395>）：量化並快取層輸入 X、即時重建 K/V——①×③ 的交叉。

---

### M 組：壓縮快取（對應精度階；**F3 的 1999 年版**）

#### M1. Douglis, The Compression Cache
- **(a)** 用部分 RAM 以壓縮形式存頁，延伸實體記憶體；自適應地在壓縮／未壓縮區間分配。
- **(b)** F. Douglis, USENIX Winter 1993, pp. 519–529. <https://www.usenix.org/conference/usenix-winter-1993-conference/compression-cache-using-line-compression-extend-physical>
- **(c)** ①。Wilson 等人事後指出其結果不一致、部分歸因於「兩區依 recency 競爭 RAM」的策略（見 M2）。
- **(d)** KV 版：AdaptCache、EvicPress（見下）。
- **(f)** 高（概念老）。

#### M2. Wilson, Kaplan, Smaragdakis, The Case for Compressed Caching
- **(a)** 壓縮快取＝記憶體階層中新增一層；用 LRU 次序（含最近被逐出者）的 miss-rate 直方圖做線上 cost/benefit，決定壓縮多少。
- **(b)** P. R. Wilson, S. F. Kaplan, Y. Smaragdakis, USENIX ATC 1999, pp. 101–116. <https://www.usenix.org/legacy/events/usenix99/full_papers/wilson/wilson.pdf>【原文，全文核對 §3、§3.1】：
  - 對 Douglis：「Given that the uncompressed cache always holds more recently-touched pages than the compressed cache, this scheme requires a bias to ensure that the compressed cache has any memory at all … a robust adaptive cache-sizing policy cannot be based solely on the LRU ordering of pages within the caches.」
  - 範例：100 frames、壓縮 2:1，50/50 配置可存 150 頁；**效益＝落在 LRU 次序第 101–150 位的 fault 數 × 磁碟成本；成本＝落在第 51 位之後的所有存取 × (解)壓縮成本**。
- **(c)** ①。**與 F3 的機制高度吻合**：純 LRU 次序下壓縮區拿不到記憶體（+0.004%），要有非 recency 的依據（F3 的「價值÷位元組」）才有效；其效益公式就是 §4 定律三的 [C, C_eff] 區間（我們的判讀）。對 F8：全域 INT4＝全部壓縮，等效容量 ×3.77，效益＝MRC 在 [C, 3.77C] 的落差。
- **(d)** 本次未找到 KV 論文引用 Wilson／Douglis（本地 35 篇 grep 0 筆）。KV 端的「壓縮＋逐出聯合」：AdaptCache（S. Feng, H. Li et al., arXiv 2509.00105 <https://arxiv.org/abs/2509.00105>）【原文，本地全文核對：MCKP、以「每單位省下空間的效用下降」（marginal utility drop）貪婪選擇壓縮或逐出】；EvicPress（S. Feng, Y. Liu et al., arXiv 2512.14946 <https://arxiv.org/abs/2512.14946>）【原文，本地全文核對 §4.4：MCKP、NP-hard、重複選「效用下降最小」的更新直到放得下，逐層遞迴】。
- **(e)** §6 步驟 2：用 MRC 直接算 Wilson 式效益／成本，對照 F3 的 +0.004% 與 −18.00%、以及 F8 的 INT4 增益——若三者都能用同一條 MRC 解釋，F3 與 F8 就是同一條定律的兩端。
- **(f)** 高（「LRU 次序讓壓縮階無效」是 1999 年的已知結論；AdaptCache／EvicPress 已有 value-per-byte 的 KV 版）。

#### M3. Linux zswap
- **(a)** swap 出去的頁先壓縮進 RAM 池，池滿時以 LRU 寫回 swap 裝置；用 CPU 換 I/O。
- **(b)** Linux kernel documentation, "zswap." <https://docs.kernel.org/admin-guide/mm/zswap.html>
- **(c)** ①②：GPU-INT4 之於 CPU／SSD ≈ zswap 之於 swap（我們的判讀）。
- **(d)** 本次未找到 KV 引用。
- **(f)** 低。

#### M4. Google Software-Defined Far Memory（zswap 為遠記憶體）
- **(a)** 以「冷頁＝超過 T 秒未存取」主動壓縮；以 promotion rate（從遠記憶體換回的速率）當 SLI、設目標值；以 Gaussian Process Bandit 自動調參。
- **(b)** A. Lagar-Cavilla et al., ASPLOS 2019. <https://doi.org/10.1145/3297858.3304053>【原文，全文核對 §4.1 cold age threshold T、promotion rate 定義】。
- **(c)** ①②：「T 秒沒用就降精度／降層，並以『升回率』SLO 控制 T」正是 TTL（I2）＋ski rental（J1）的產品化版本；可直接移植成 Tiara 的線上控制器（我們的判讀）。
- **(d)** 本次未找到。
- **(e)** 以「降級後被升回的比例」當 SLI，掃 T，畫升回率 vs 總時間。
- **(f)** 低–中。

#### M5. Meta TMO（Transparent Memory Offloading）
- **(a)** 以 PSI（pressure stall information）量測記憶體不足造成的實際停頓，userspace 代理 Senpai 依壓力回饋調整回收量；後端為壓縮記憶體或 SSD swap。
- **(b)** J. Weiner et al., ASPLOS 2022 <https://doi.org/10.1145/3503222.3507731>；Meta 工程部落格（控制式 `reclaim = current_mem × reclaim_ratio × max(0, 1 − psi_some/psi_threshold)`；後端「依應用的可壓縮性手動選壓縮記憶體或 SSD」）<https://engineering.fb.com/2022/06/20/data-infrastructure/transparent-memory-offloading-more-memory-at-a-fraction-of-the-cost-and-power/>（部落格全文核對；論文摘要核對）。
- **(c)** ①②：TMO「壓縮 vs SSD」是人工選的——Tiara 若能以 κ 與 ε 自動選，是有意義的差異（我們的判讀）；「以實際停頓而非 miss 數做回饋」與 Baleen、delayed hits 同精神。
- **(d)** 本次未找到。
- **(f)** 低–中。

---

### N 組：資料庫——物化 vs 重算、成本導向緩衝

#### N1. 物化視圖選擇（HRU data cube）
- **(a)** 視圖間有依賴格（lattice），貪婪地選「在已選集合下帶來最大改善」的視圖。
- **(b)** V. Harinarayan, A. Rajaraman, J. D. Ullman, "Implementing Data Cubes Efficiently," SIGMOD 1996（Best Paper）<https://doi.org/10.1145/235968.233333>【依搜尋摘要與課程投影片：貪婪收益 ≥ (1−1/e)≈63% 最優；原文 PDF 未逐字核對】。
- **(c)** 聯合：「存哪些 KV（物化）、哪些要用時重算」的資料庫原型。**提醒**：HRU 的保證依賴收益的次模性；prefix 相依造成互補，不能直接套（我們的判讀）。
- **(d)** 本次未找到。
- **(f)** 中。

#### N2. WATCHMAN
- **(a)** 資料倉儲查詢結果快取，准入與替換用「profit」＝參照率 × 查詢執行成本 ÷ 大小。
- **(b)** P. Scheuermann, J. Shim, R. Vingralek, VLDB 1996, pp. 51–62 <https://dblp.uni-trier.de/rec/conf/vldb/ScheuermannSV96.html>（摘要核對）。
- **(c)** ③：F3 的價值密度在資料庫的前身。
- **(f)** 高（概念老）。

#### N3. Recycler（MonetDB 中間結果回收）
- M. Ivanova, M. L. Kersten, N. J. Nes, R. Gonçalves, SIGMOD 2009 <https://doi.org/10.1145/1559845.1559879>（*ACM TODS* 版 <https://dl.acm.org/doi/10.1145/1862919.1862921>）。(c) 快取中間結果 vs 重算。

#### N4. Nectar（資料中心的衍生資料與計算）
- **(a)** 衍生資料集以產生它的程式為身分；可刪、可重算；以成本效益比刪除價值最低者；輸入也被刪時遞迴回溯重算。
- **(b)** P. K. Gunda, L. Ravindranath, C. A. Thekkath, Y. Yu, L. Zhuang, OSDI 2010 <https://www.usenix.org/conference/osdi10/nectar-automatic-management-data-and-computation-datacenters>【原文，全文核對：`CBRatio = (S × ΔT)/(N × M)`，刪除比值最大者；新項目以 lease 保護】。
- **(c)** ③：`CBRatio` 的倒數就是「(使用次數×重算機器時間)/(大小×閒置時間)」＝DTR／LHD 的價值密度；「遞迴回溯」＝KV 重算鏈（我們的判讀）。
- **(f)** 高（概念老）。

#### N5. MCKP、LP-dominance 與樹背包
- **(a)** 多選一背包（每組選一個選項）；LP 鬆弛可用 dominance 規則與凸包上的貪婪求解；樹上的偏序背包（選子必選父）可用 DP 得偽多項式解與 FPTAS。
- **(b)** G. B. Dantzig, "Discrete-Variable Extremum Problems," *Operations Research* 5(2):266–288, 1957 <https://doi.org/10.1287/opre.5.2.266>（連續背包依價重比貪婪）；P. Sinha, A. A. Zoltners, "The Multiple-Choice Knapsack Problem," *Operations Research* 27(3):503–515, 1979 <https://doi.org/10.1287/opre.27.3.503>（兩條 dominance 規則給 LP 上界）；D. S. Johnson, K. A. Niemi, "On Knapsacks, Partitions, and a New Dynamic Programming Technique for Trees," *Math. of OR* 8(1):1–14, 1983 <https://doi.org/10.1287/moor.8.1.1>（皆摘要核對）。
- **(c)** ①②聯合。每個 block 的選項 {BF16, INT4, CPU, SSD, DROP} 是一組 MCKP 選項，在「(GPU 位元組, 預期成本)」平面上**不在下凸包上的選項（LP-dominated）LP 解永遠不選**——這給 F3 一個乾淨的判準：INT8 只有在其（dequant＋品質懲罰）每省一位元組的代價低於下一個選項時才會被選；LRU 次序根本不做這個比較（我們的推導，非文獻）。樹背包則是「prefix 樹上選哪些節點留 GPU」的靜態鬆弛（快照 Oracle）。
- **(d)** AdaptCache、EvicPress、KVDrive（J. Lin et al., arXiv 2605.18071 <https://arxiv.org/abs/2605.18071>）都把問題寫成 MCKP（本地全文核對）。
- **(e)** 對每個 (model, P 分箱) 畫選項點與下凸包，列出哪些精度／層級在凸包上——不跑模擬就知道哪些動作是廢的。
- **(f)** 中。

---

### O 組：排隊與排程

#### O1. M/G/1 與 Pollaczek–Khinchine（服務時間變異的排隊懲罰）
- **(a)** M/G/1 的平均等待正比於服務時間的二階矩 E[S²]，不只是平均。
- **(b)** 教科書：M. Harchol-Balter, *Performance Modeling and Design of Computer Systems: Queueing Theory in Action*, Cambridge University Press, 2013（ISBN 978-1-107-02750-3）<https://assets.cambridge.org/97811070/27503/frontmatter/9781107027503_frontmatter.pdf>（目錄核對；M/G/1 章節未逐頁核對）。
- **(c)** ②：F11 的 SSD warm 成本 spread 90.1% → 即使平均成本相同，SSD 階也會因 E[S²] 大而放大排隊延遲；放置決策應看二階矩（我們的判讀）。VA-CDH（B1）在 delayed hits 中也納入變異。
- **(e)** 在模擬器中把 SSD 成本改為實測分佈抽樣（而非中位數），比較 TTFT 平均與尾端。
- **(f)** 低。

#### O2. SRPT
- L. Schrage, "A Proof of the Optimality of the Shortest Remaining Processing Time Discipline," *Operations Research* 16(3):687–690, 1968 <https://doi.org/10.1287/opre.16.3.687>。(c) 放置改變每個請求的 prefill 服務時間，進而影響 SRPT 類排程的效益；Continuum 原文以 program-level FCFS 近似 SRTF。

#### O3. PASTA
- R. W. Wolff, "Poisson Arrivals See Time Averages," *Operations Research* 30(2):223–231, 1982 <https://doi.org/10.1287/opre.30.2.223>。(c) 理論模型用「時間平均的駐留率」換算「請求看到的命中率」只在 Poisson 到達時成立；Mooncake 若是突發的（agent／tool call），I1／I2 的 Che／TTL 近似可能偏差（我們的判讀）。(e) 先算每個 block 的到達間隔變異係數與 burstiness，再決定能否用 Che 近似。

#### O4. LLM 推論的排隊／排程理論（對放置是背景）
- M. Mitzenmacher, R. Shahout, "Queueing, Predictions, and LLMs: Challenges and Open Problems," arXiv 2503.07545 <https://arxiv.org/abs/2503.07545>（*Stochastic Systems* 版 <https://pubsonline.informs.org/doi/10.1287/stsy.2025.0106>）；J. G. Dai, T. Deng, Y. Li, T. Peng, "Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents," arXiv 2504.07347 <https://arxiv.org/abs/2504.07347>；C. Nie, N. Si, Z. Zhou, "A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints," ICML 2026, arXiv 2605.04595 <https://arxiv.org/abs/2605.04595>；Jaillet et al.（J6）。（皆摘要核對）
- **(f)** 對放置：低（他們處理的是排程與記憶體穩定性，不是分層放置）。

---

## §3 回應協調者的三個新問題

### 3.1 F8「量化與放置是替代品」——哪個經典理論能解釋、甚至事先預測？

**答：Mattson 的 stack distance／MRC（I5）＋ compulsory-miss 下限，加上 Wilson 1999 的壓縮快取 cost/benefit（M2）；rent-or-buy（J2）解釋「為什麼連搬運那一半也被吃掉」。**

1. **MRC 的 inclusion 性質**（Mattson 1970；OPT 為 stack algorithm 一點本文未逐字核對原文）：LRU 與 OPT 的 miss 數都隨容量單調不增；兩者在容量 ≥ footprint 時都只剩 compulsory miss（第一次觸碰，沒有策略能避免）。所以「放置策略相對 LRU 的 headroom」在任何容量 C 都受 `MR_LRU(C) − MR_OPT(C)` 限制，且這個間隙在 C→0 與 C→footprint 兩端都趨近 0，只在 MRC 的轉折區（懸崖區，I6）最大（我們的推導，非文獻；教科書性質的直接推論。此處是單層、以 miss 計；多層與成本加權時，改用 Gill 的聚合容量（F4）與 weighted OPT（H4））。
2. **INT4 = 沿 MRC 右移 3.77 倍**：F8 的容量倍數 3.77× 使等效容量 C→3.77C。若 3.77C 已落在 MRC 平坦區，則 (i) LRU 的 miss 已接近 compulsory 下限 → INT4+LRU 拿走幾乎全部 headroom（F8：91.8%／103.9%）；(ii) 在 3.77C 處 LRU–OPT 間隙很小 → 殘餘 headroom 小（F8：2.02%／3.52%）。103.9% > 100% 也可解釋：BF16 的 Oracle 被限制在容量 C，而 INT4+LRU 在 3.77C 上，容量優勢勝過策略優勢（我們的判讀）。
3. **為什麼兩個 Oracle 幾乎一樣（−0.08%）**：兩者都接近「成本下限」——新 token 的 prefill 是 compulsory、無法避免的計算；BF16 Oracle 靠「少重算」（把 capacity miss 轉成 CPU 取回），INT4 靠「多裝」（直接消掉 capacity miss），抵達同一地板（我們的判讀，與 remote 原文敘述一致）。
4. **rent-or-buy 視角**（J2、§4 定律二）：INT4 同時 (a) 把每個 block 的 GPU 租金降為 1/3.77（容量約束放鬆 → Lagrange 乘數 λ 下降）、(b) 若搬運時間也 ÷3.77，又降低了「載回」的喚醒成本——放置策略的兩個主要槓桿同時被削弱，剩下能做的只有在少數仍處於斷點附近的 block（我們的判讀）。
   - ⚠️ **與 F9 的一致性要核對**：EXPERIMENTS 原文寫「CPU/SSD 搬運 ÷3.77 +dequant」且「全部來自實測」。但依 F9，32–64 KiB 描述符時每筆時間幾乎全是固定成本 t0（例如 32 KiB 時純傳輸只佔約 0.57 µs，而 t0 約 13 µs（原文 12.9 µs），算術，非實測），若 INT4 只是把每層頁縮小、描述符數不變，按位元組 ÷3.77 會高估 INT4 的搬運收益。反過來，表 2.2 的 ≤16 KiB 列隱含每筆只有約 2.7–2.8 µs（見 3.2），而 32 KiB÷3.77≈8.5 KiB 正好跨進那個 regime，實際縮放可能接近、也可能不同於 3.77。**建議核對 S1 中 INT4 搬運常數是 INT4 下的實測搬運時間，還是按位元組比例縮放的**（CLAUDE.md 規則 1、3）。
5. **能事先預測嗎？能**（這正是使用者要的「先做理論模型」）：
   - 只用 trace 算兩條 MRC（成本加權的 LRU 與 OPT），在 C 與 3.77C 兩處讀間隙，即可預測 BF16 headroom 與 INT4 殘餘 headroom，**再**與 F8 的 20.66%／2.02% 比對——若吻合，替代關係就被一條教科書定律解釋（這對論文是雙面刃：解釋力強，但新意變成「量化 KV 並以 bit-width 位移 MRC」）。
   - **預測 remote S2 的 q 掃描形狀**：若 q 比例的 block 以 INT4 存，等效容量倍數 `m(q) = 1 / (1 − q·(1 − 1/3.77))`（我們的推導，非文獻；由「(1−q)·s + q·s/3.77 的平均佔用」直接得到；m(1)=3.77，m(0.5)≈1.58，算術）。循序（S1）的 headroom 曲線 ≈ MRC 間隙在 m(q)·C 的值；聯合（S2）只能在「選對哪些 block 壓縮」上贏過 S1——依 Wilson 的分析，最值得壓縮的是 stack distance 落在 (C_uncompressed, C_eff] 的 block；若品質懲罰對所有 block 都一樣，聯合與循序的差距主要取決於 S1 的挑選規則（AdaptCache 式 LFU）與這個理想集合的重疊程度（我們的判讀）。因此 **S2−S1 的真正來源是「逐 block 品質敏感度的異質性」**，而這需要逐位置／逐層的 ε——與 remote 的「先把 ε 量準」結論一致。
6. **這也同時解釋 F3**：3090 的 INT8 +0.004% 與 MI300X 的 INT4 大增益是同一條定律的兩端——增益取決於「實際以壓縮形式常駐的位元組 U_c」，不是「有沒有精度階」（§4 定律三）。
7. **旁證**：Tumkur et al.（arXiv 2609.16215，§7）摘要稱分層的收益主要來自各層容量比例而非放置策略；Ganjihal 2026（K11）稱 recency 已近 OPT——兩者都與「容量主導、策略次要」一致。
8. **新穎性警告**：在快取社群眼中，「放大容量後 LRU≈OPT」是可預期的；可守的增量是 (i) 以 **bit-width** 作為容量位移的旋鈕並在 KV 上量化、(ii) 以**延遲／成本**（含 F1 位置項）而非 hit ratio 為單位、(iii) 給出**夾住 OPT 的上下界**（H3／H4），而不是只用貪婪 Oracle。

### 3.2 F9 是 Little's Law 的實例嗎？與「一句話定律」一致嗎？

**是，而且可以用它做單位與一致性的交叉驗算（CLAUDE.md 規則 6 的精神）。**

- **模型**：每筆描述符延遲 `L(S) = t0 + S/B`，併發 C 時有效頻寬 `X = C·S/L(S)`（Little；Strata §3.1 的同一條式子）。C=1 時 `t0 = S·(1/X − 1/B)`。
- **以 EXPERIMENTS 表 2.2 的數字驗算（算術，非實測；B = 57.4 GB/s，KiB/MiB 以 1024 為底、GB 以 10⁹）**：
  - 32 KiB、2.4 GB/s → t0 ≈ 13.1 µs（原文 12.9 µs，差異在 X 四捨五入的範圍內）；64 KiB → ≈12.8 µs；128 KiB → ≈13.0 µs；256 KiB → ≈13.3 µs；1 MiB → ≈13.3 µs；4 MiB → ≈13.4 µs：**≥32 KiB 各列一致指向 t0 ≈ 13 µs，符合「與 payload 無關的固定成本」**。
  - **1.5 MiB、38.9 GB/s → t0 ≈ 13.0 µs，而非原文的 16.3 µs**。可能原因：推算時用了不同的 B、不同的描述符數（表中 1 MiB 與 1.5 MiB 都寫 256 筆）或 GB／GiB 混用。**建議回原始 log 重算 16.3 µs 的來源。**
  - **8 KiB（2.8 GB/s）與 16 KiB（5.5 GB/s）→ t0 只有約 2.7–2.8 µs**，且 16 KiB 比 32 KiB 還快（非單調）。在 Little 的形式下這有兩種解讀：小描述符走不同路徑（t0 較小），或同時有約 4–5 個在飛（若 t0 仍為 13 µs，則 C≈4.5）。**只看吞吐無法分辨 C 與 t0**（Little 的本質），需同時量每筆延遲或在飛數。這個 regime 切換也直接影響 3.1 第 4 點（INT4 頁縮到約 8.5 KiB）。
- **與 §4 定律的一致性**：
  - 定律一（Amdahl）：一致。F10 的「MI300X 端到端 headroom 3.76–4.41%」與「prefill 口徑 headroom × prefill 佔比」同量級：20.66% × (1−0.82)…(1−0.78) ≈ 3.7%–4.5%（算術，非實測；兩者的 workload 未必逐一對應）。
  - 定律二（KV 五分鐘法則／lower envelope）：一致，但**必須**把每一層的存取成本寫成仿射式 `a_t(b) = n_desc(b)·t0 + s_b/B_t (+ dequant)`；於是 κ 與 T* 取決於描述符粒度 n_desc，而不只是硬體頻寬——這正是 remote §2.4「κ 不是硬體性質」的形式化（我們的推導，非文獻）。也由此可做 what-if：把 n_desc 合併（例如到 4 MiB）後重算 lower envelope，預測哪些狀態會從「被支配」變成「在凸包上」——這回答「修掉描述符問題後放置結論會不會翻轉」，且不必先實作合併。
  - 定律三（壓縮等效容量）：一致；但 INT4 對「載回」成本的縮放要依上面的仿射式重算，不能用位元組比例。
- **新穎性提醒**：vLLM（SOSP'23 §7.3）已寫「小 block → 大量小傳輸 → 有效 PCIe 頻寬受限」；Strata §3.1 已用 Little 推「要放大 S」；Tutti（S. Qiu et al., arXiv 2605.03375 <https://arxiv.org/abs/2605.03375>）也指出 GDS 仍需 CPU 逐 I/O 介入、軟體開銷大（本地 `tutti2026.txt` L372–374）。F9 的新意在「ROCm 上 Triton 快速路徑被停用、`hipMemcpyBatchAsync` 以 `numAttrs=0` 呼叫」這條具體的實作因果鏈、以及 t0 的量測值——不在「小傳輸很慢」本身。

### 3.3 F10／F11 對 learning-augmented 條目的直接證據

- **F10 是「沒有 robustness 的學習策略」的教科書案例**：learning-augmented 文獻的核心要求是 consistency（預測好時接近 OPT）與 robustness（預測壞時不比經典演算法差太多）同時成立（K1–K7）。GBDT 的 AUC 0.917–0.922、ECE 0.003–0.004 說明預測器本身「會不會再用」預測得好，但策略在 MI300X 輸給 tier_fs 9.7–47.3%——代表 (i) 策略沒有保底、(ii) 預測目標與決策需求不一致（我們的判讀）。
- **為什麼同程式碼在 3090 +81.85%、MI300X 卻輸**（我們的判讀，三條可檢驗的假說）：
  1. **可取得的 headroom 太小**（A2＋K11）：MI300X 上 baseline 間只差 1.6%、decode 佔 78–82%，任何誤判的代價都大於可能的收益。
  2. **決策門檻是依 3090 的 κ 校準的**（J2／§4 定律二）：斷點 T* 與 P* 隨 κ 移動；κ 在 MI300X 上由描述符固定成本主導（F9）、κ_ssd=0.72（F11），3090 上學到的門檻在 MI300X 落在錯的位置。
  3. **分佈偏移**（K8 的 DAgger 觀點）：訓練資料來自某個策略造出的狀態分佈，換硬體後策略的行為改變，狀態分佈也變了。
- **F11（query 側 AUC 0.480–0.500）＝「資訊量為零的預測器」**：在 LV18 等框架中，這相當於預測誤差 η 最大，consistency 的好處完全消失，只剩 robustness 有意義；盲目跟隨（BlindOracle）的最壞界是無界的，所以必須有 LRU／tier_fs 保底（K3、K4）。F11 也說明「注意力由位置決定」→ 位置應當是 LHD 式的**類別鍵**（C5），而不是讓模型去學的特徵（我們的判讀）。
- **可直接做的三個便宜實驗**（皆為提案）：(a) K4 的「GBDT ⊕ tier_fs」切換包裝，兩個平台各跑一次；(b) 以 LRB 的 good decision ratio 與 Kendall τ 取代 AUC 重新評估 GBDT（C6、K8）；(c) 依 LAH（K10）把學習縮成「每平台學 λ 與 P* 修正」兩個參數，檢查跨平台可移植性。
- **威脅**：LCR／LARU（K9）已把 robust learning-augmented LRU 用於 LLM；Ganjihal 2026（K11）與 Tumkur et al.（§7）主張 recency／容量已足夠。Tiara 若要保留學習元件，可守的位置是「成本異質（F1 位置項、F2／F9 的層級成本）下的學習」，不是 hit-ratio 預測。

---

## §4 Tiara 的一句話定律候選（**我們的推導，非文獻**）

### 定律一：放置的 Amdahl 上限

**式子**：令端到端時間 `T = T_pf + T_dec (+ T_other)`，`f_d = T_dec / T`。若放置只影響 prefill（F5），則任何策略的端到端相對改善
`r_e2e = (T_pf/T) · r_pf ≤ (1 − f_d) · r_pf ≤ 1 − f_d`，端到端加速 `≤ 1 / f_d`。

**推導**：Amdahl（A2）直接代入；`T_other` 計入「非 decode」會讓界更鬆（仍是上界）。

**前提**：(i) decode 不受放置影響——若 GPU 上以 INT4 存 KV 且 decode attention 直接讀低精度，或 continuous batching 下 prefill 負載改變 decode 的 TBT，則不成立（A3）；(ii) f_d 以同一 workload 的同一口徑量（每請求關鍵路徑或 GPU 時間佔比，要寫清楚）。

**對 M4 的含義（算術，非實測）**：
- 平台 A：f_d ∈ [48.8%, 75%] → prefill 佔比 ≤ [25%, 51.2%]；端到端 ≥15%（GO）需要 prefill headroom ≥ 15%/0.512 ≈ 29.3% 到 15%/0.25 = 60%；端到端 ≥5% 需要 ≥ 9.8%–20%；端到端加速上限 1/0.75 ≈ 1.33× 到 1/0.488 ≈ 2.05×。
- 平台 B：f_d ∈ [78%, 82%]（F10）→ prefill 佔比 ≤ [18%, 22%]；端到端 ≥15% 需要 prefill headroom ≥ 68.2%–83.3%。F10 的端到端 3.76–4.41% 與 20.66%（prefill 口徑）×0.18–0.22 ≈ 3.7%–4.5% 同量級（算術）。
- ⇒ **M4 的 GO/NO-GO 必須先寫明是哪個口徑**；在平台 B 端到端口徑下，15% 門檻幾乎不可能達到。

### 定律二：KV 的五分鐘法則＝lower envelope（含 Little 截距）

**式子**：對 block b（GPU 位元組 s_b、位置 P_b），在閒置 Δ 後被重用，把每個狀態 t 寫成 (租金率 h_t, 喚醒成本 a_t)：

| 狀態 t | h_t（GPU 租金率） | a_t（要用時的成本） |
|---|---|---|
| GPU-BF16 | λ·s_b | 0 |
| GPU-INT4 | λ·s_b/3.77 | dequant＋μ·品質懲罰 |
| CPU | ≈0 | n_desc·t0 + s_b/B_cpu |
| SSD | ≈0 | C_ssd（F2：與 P 無關；約 65% 為軟體） |
| DROP | 0 | C_rc0 + α·P_b（F1；前綴也被丟時加上整條鏈，L2） |

離線最優 `LE_b(Δ) = min_t { h_t·Δ + a_t }`；**一句話**：「block 只在預期重用間隔短於 T* 時才值得佔 GPU」。只比較 GPU-BF16 與零租金狀態（CPU／SSD／DROP）時 `T*_b = min_{t∈{CPU,SSD,DROP}} a_t / (λ·s_b)`；若 GPU-INT4 也在 lower envelope 上，BF16→INT4 的斷點為 `a_INT4 / (λ·s_b·(1 − 1/3.77))`，INT4→零租金狀態的斷點為 `(min a_zero − a_INT4) / (λ·s_b/3.77)`（前者須小於後者，INT4 才在凸包上）。更一般地，**不在 lower envelope 上的狀態永遠不該被用**。

**推導**：(1) 把 GPU 容量約束以 Lagrange 乘數 λ 鬆弛（I4：Dehghan et al. 證明 characteristic time 對應此乘數）→ 各 block 獨立；(2) 每個 block 的單段閒置期即多態 rent-or-buy，離線最優為 lower envelope（J2：Irani et al.）；(3) 線上在斷點轉換狀態 → 對每段閒置期 2-competitive（J2 原文結果；前提是 block 間獨立，容量耦合由 λ 近似）；(4) λ 以「預期 GPU 佔用＝C_GPU」的定點求得（I1 的 Che 定點，或 A1 的 `W_GPU = C_GPU/λ_B`）；(5) 品質約束 ε 以第二個乘數 μ 同法處理。

**推論 A（P* 是它的特例）**：只比較兩個 h≈0 的狀態 SSD 與 DROP 時，DROP 在 `P < P* = (C_ssd − C_rc0)/α` 時較便宜（F2）。把 F2 的軟體成本拆開 `C_ssd = C_io + C_sw` 得 `P* = (C_io + C_sw − C_rc0)/α`：移除軟體成本會讓 P* 左移 C_sw/α 個 token。**一致性檢查建議**：用 F1 的同一組常數驗算 `C_ssd = C_rc0 + α·P*` 是否等於實測 C_ssd；若不等，代表 P* 不是由這條線性擬合算出（不同 chunk 大小或另有常數），論文中兩者不能混用。

**推論 B（Cake 的會合點不是 P*）**：若計算與 I/O 可同時進行（Cake），單請求的最短延遲不是「每個 chunk 選較便宜者」，而是讓「從前面重算的累計時間」等於「從後面載入的累計時間」的會合點 m*：`Σ_{i<m*} (C_rc0 + α·P_i) ≈ (N − m*)·a_load`。P* 適用於「資源成本／吞吐」（兩種資源都有其他工作可做時），m* 適用於「單請求延遲」（資源閒置時）——Tiara 必須先決定目標是哪一個。

**推論 C（被支配的狀態）**：F11 的 κ_ssd<1 表示在那些位置 a_SSD > a_DROP 且兩者 h≈0 → SSD 不在 lower envelope 上 → 最佳策略在這些位置永遠不用 SSD；FP8／INT8 若 dequant 與品質懲罰不顯著小於 INT4 的替代，也可能被支配（N5 的 LP-dominance 判準）。

**與 F9 一致**：a_CPU、a_SSD 含 n_desc·t0 截距，所以 T*、P*、哪些狀態在凸包上，都會隨描述符粒度而變——「κ 是實作性質」在此式中是自然結果。

**前提**：降級本身的成本（D2H 複製、量化運算）要併入 a_t 或假設可被重疊隱藏——Irani et al. 的模型只對「喚醒」收費；block 大小固定（全 BF16 或全 INT4）；重用間隔分佈穩態；容量耦合由 λ 近似（真實系統有突發，λ 需線上調，M4／M5 的 SDFM／TMO 是可參考的控制器）；delayed hit（B1）另計。

**新穎性**：MatKV 的 ten-day rule 只有兩態、美元單位；InferCept／Continuum 是已知區間長度的特例；本次搜尋未見把 KV 多層寫成多態 lower envelope 並給 competitive 保證者（中風險）。

### 定律三：壓縮階的等效容量律（替代律）

**式子**：在實體容量 C（位元組）中，若以壓縮比 r 常駐的 block 在未壓縮下共 U_c 位元組，則
`C_eff = C + (1 − 1/r) · U_c`，
壓縮帶來的 hit 增量 `≈ Σ_{d ∈ (C, C_eff]} h(d)`（h 為 stack distance 直方圖），
而**任何放置策略相對 LRU 的剩餘 headroom ≤（成本加權的）`MR_LRU(C_eff) − MR_OPT(C_eff)`**。

**推導**：Mattson 的 stack inclusion（I5）＋Wilson 1999 的區間效益（M2）；U_c 由「壓縮後 block 的保留機制」決定：
- LRU 次序下降級最冷者（F3）：壓縮段是過渡狀態，U_c≈一個 block → C_eff≈C → 增益≈0（對應 +0.004%）。
- 價值÷位元組次序（F3）：壓縮後價值密度升高而被保留，U_c 變大 → 沿 MRC 右移（對應 −18.00%）。
- 全域 INT4（F8）：U_c＝全部，C_eff＝3.77C → 若已近 footprint，LRU≈OPT≈compulsory，剩餘 headroom 小（對應 2.02%／3.52%）。
- 品質預算 q：`C_eff/C = m(q) = 1/(1 − q(1 − 1/r))`，r=3.77（見 3.1）。

**一句話**：「量化值多少，取決於它把你推到 MRC 的哪裡；推到平坦區，放置就沒事可做。」

**前提**：stack inclusion 只對 LRU／OPT 這類 stack algorithm 嚴格成立；價值÷位元組不是 stack algorithm，只能用 miniature simulation（I5）近似；成本加權 MRC 需要每次 miss 的成本（F1、F2 已有）；解壓（dequant）與品質懲罰要從增益中扣除。

**新穎性**：原理是教科書（Mattson 1970、Wilson 1999）；KV 端 AdaptCache 已展示壓縮提高命中率（remote `PAPERS_EXPLAINED_20260920.md` 的核對）。可守的是「以 bit-width 當 MRC 位移、以成本／延遲為單位、並以上下界夾住 OPT」的量化證據（中風險）。

---

## §5 對 Tiara 最有用的前 8 項（兼顧原 Tiara 故事與 remote 的兩個新方向：(A) 量化與放置是替代品、(B) 有效頻寬是實作性質）

1. **Stack distance／MRC＋Wilson 壓縮快取（I5、M2）**——唯一能「事先預測」F8 替代關係、同時解釋 F3 的工具；一次 trace 掃描就能畫出來，成本極低；直接支撐方向 (A) 的主圖（headroom 隨 bit-width 的變化＝MRC 間隙隨 m(q) 的變化）。風險：原理是教科書，要把新意放在 KV 的量化證據。
2. **離線上下界工具組（H3 FOO、H4 weighted caching min-cost flow、F4 Gill OPT-UB/LB、H5 NP-hard）**——F6 說 Oracle 是貪婪解；F8 的「殘餘 2.02%」因此只是真 headroom 的下界（前提：貪婪 Oracle 是同一模型下的可行排程）。沒有 OPT 成本的下界，替代主張（A）與 M4 判定都站不穩。全 BF16／全 INT4 時 block 等大小，單層 GPU 的最優可用 min-cost flow **精確**求出，這是最便宜、最有說服力的一步。
3. **多態 ski rental／lower envelope（J2，含 J1、J3、K7 與 NeurIPS'21 學習版）**——把 ②③（加 ① 當作租金折扣）統一成一個逐 block 模型，給出可跨硬體重新推導的門檻（T*、P*）、2-competitive 的線上策略、以及「被支配狀態」的預測（F11 的 κ_ssd<1 → SSD 不該被用）。是 §4 定律二的數學本體。
4. **Little's Law（A1）**——F9 的正確框架（`X = C·S/(t0 + S/B)`），並讓「κ 是實作性質」變成可檢驗的式子；還能反查資料一致性（§3.2 的 1.5 MiB 與 ≤16 KiB 異常）。風險高（Strata、vLLM 已用），當工具用，不當貢獻。
5. **Amdahl（A2）**——一行字就把 M4 的口徑問題與 MI300X 的端到端天花板講清楚（§4 定律一）；審稿人一定接受。
6. **Learning-augmented caching（K3 Wei、K4 Chłędowski、K7 Purohit、K9 LCR、K10 LAH）＋LRB 的 Belady boundary（C6）與排序指標（K8）**——直接解釋 F4、F10、F11，並給出便宜可做的修法（切換保底、改用 good decision ratio／Kendall τ、只學少數參數）。風險：LCR 已用於 LLM。
7. **五分鐘法則／10-byte rule（A4）**——最好講的故事句：「KV 的五分鐘法則由 κ 決定，而 κ 會隨硬體與實作移動」；與定律二一體兩面。風險：MatKV 已把它帶進 KV（但只有兩態、美元單位）。
8. **價值密度家族（C1–C3 GreedyDual／GDS／GDSF、C5 LHD、L2 DTR、N4 Nectar）**——說明 F3 的「剩餘價值÷位元組」為何有效，並提供必須比較的強 baseline（尤其 RAGCache 的 PGDSF 與 DTR 的前綴鏈成本）。風險高：這條路已擁擠，只能當對照。

**次要但值得一提**：Che／TTL／效用-Lagrange（I1、I2、I4）是定律二求 λ 的方法；delayed hits（B1）決定模擬器計時是否正確；inclusive／PROMOTE（D4、F4）與 SSD 准入（D1、D5）影響 SSD 寫入量；M/G/1 的二階矩（O1）說明 F11 的 SSD 90.1% 變異有排隊懲罰；κ_crit（A3）是必須正面回應的威脅。

---

## §6 「先做理論模型」的最小路線（全部是提案，未執行；每步都有可證偽的輸出）

> 依 CLAUDE.md §4：每一步都要有 run_id、cmd、輸出檔；trace 載入時依規則 6 以請求長度驗算 `hash_ids`＝512-token block。

1. **步驟 0｜trace 前處理與單位斷言**：把 Mooncake 轉成 block 級請求序列（時間戳、block id、位置 P＝index×512，若模擬器用其他 block 大小則換算並斷言一致）。輸出：block 請求數、distinct block 數（footprint）、每 block 的到達間隔 CV（O3，決定能否用 Che 近似）。
2. **步驟 1｜Little 一致性（A1）**：在既有模擬器 run 上驗 `mean(B) ≈ mean(λ_B)·mean(W)`，得 W_GPU。
3. **步驟 2｜MRC（I5、M2）**：Fenwick tree 算 LRU stack distance → MRC_LRU；在容量格點跑 Belady → MRC_OPT；各做成本加權版（miss 乘 min(C_ssd, C_rc0+α·P, a_CPU)）。標出 C_BF16、3.77·C_BF16、footprint。**預測**：C 處的間隙 ≈ BF16 headroom、3.77C 處的間隙 ≈ INT4 殘餘 headroom；**再**與 F8 的 20.66%／2.02%（toolagent）比對。不吻合＝MRC 模型漏了什麼（多層、delayed hit、位置成本），本身就是發現。
4. **步驟 3｜lower envelope（J2、I4、§4 定律二）**：以實測常數（a_CPU 用 `n_desc·t0 + s/B` 的仿射式）為每個 P 分箱畫 LE(Δ)，二分搜 λ 使預期 GPU 佔用＝C_GPU，得每個 block 的最佳 TTL 與層級，**不跑模擬即得預測總成本**；列出被支配的狀態（例如 κ_ssd<1 區段的 SSD）。what-if：把 n_desc 合併到 4 MiB 重算，看哪些狀態翻上凸包。
5. **步驟 4｜Oracle 夾擠（H3、H4、F4）**：全 BF16 與全 INT4 各做一次單層 GPU 的 weighted caching min-cost flow（精確）＋多層 FOO 式 LP 鬆弛（下界）；貪婪 Oracle 為上界 → 報「真 headroom ∈ [下界, 上界]」。這一步決定 F8 的替代主張與 M4 判定能否成立。
6. **步驟 5｜預測器重評（C6、K8、K4）**：對 GBDT 算 good decision ratio 與 Kendall τ；實作「GBDT ⊕ tier_fs」切換，3090 與 MI300X 常數各跑一次。
7. **步驟 6｜Amdahl 對帳（A2、§4 定律一）**：每個主結果同時報 prefill 口徑與端到端口徑，並檢驗 `r_e2e ≈ (1−f_d)·r_pf`。
8. **步驟 7｜S2 的理論預測**：用步驟 2 的 MRC 與 m(q) 畫循序（S1）的預測曲線；只有在 ε 逐 block 的異質性量到之後，聯合（S2）才有意義（與 remote 結論一致）。

---

## §7 新威脅文獻（2025–2026）與「護城河」修正

**🔴 護城河修正**：`remote/PAPERS_EXPLAINED_20260920.md` §5.1 的表把「Belady／offline-optimal 上界」列為「整個領域沒有一篇有」。**這與本次查證不符**：
- **Bidaw（FAST'26）§3.3.2、Fig. 13**：「The optimal eviction strategy—Belady's algorithm … To establish an upper bound, we simulate the hit rate of an optimal strategy by providing it with the future access trace」（本地 `bidaw2026.txt` L855–862 全文核對）——KV 逐出的 Belady hit-rate 上界已存在。（remote 同檔 L912 註明 Strata／Bidaw「這輪沒重查」。）
- **Ganjihal 2026（K11）**：在 prefix-block 模擬器計算 Belady 最優（hit ratio）。
- 另，Tiara 自己的 Oracle 是貪婪解（F6），並非 Belady，也未證明是上界；表中「👉 你：✅ Belady」需改寫。
- **仍可能空著的是 ④**：「以延遲／成本為單位、多層、位置相依成本下的 offline-optimal headroom，作為 bit-width 的函數」——本次搜尋未找到，但搜尋不完備（規則 7）。建議把護城河縮成這一句，並用 §6 步驟 4 的上下界支撐。

**2025–2026 需要在 related work 正面處理的論文**（皆已查到原始頁）：

| 論文 | 與 Tiara 的衝突點 |
|---|---|
| Meng, Lee, Wang, arXiv 2601.19910 <https://arxiv.org/abs/2601.19910>（κ_crit） | 「κ 跨硬體」的封閉式已被佔（A3） |
| Shin et al., MatKV, arXiv 2512.22195 <https://arxiv.org/abs/2512.22195>（ICDE'26） | 五分鐘法則→ten-day rule（A4） |
| Abhyankar et al., InferCept, ICML 2024 <https://proceedings.mlr.press/v235/abhyankar24a.html> | 三態 min-waste＝已知區間長度的 rent-or-buy（L8、J1） |
| Li et al., Continuum, arXiv 2511.02230 <https://arxiv.org/abs/2511.02230> | 依成本模型設 KV TTL（J1、I2） |
| Shen, Sen, Tanaka, CacheOPT, arXiv 2503.13773 <https://arxiv.org/abs/2503.13773> | swap vs recompute 的 sweet spot（P* 的請求級版本） |
| Kwon et al., vLLM, SOSP 2023 §7.3 <https://arxiv.org/abs/2309.06180> | swap/recompute 依頻寬與算力而變；小 block 壓低有效頻寬（L8、F9） |
| Jin et al., RAGCache（arXiv 2404.12457 <https://arxiv.org/abs/2404.12457>；ACM TOCS <https://doi.org/10.1145/3768628>） | prefix-aware GDSF（C3） |
| Wang et al., KVCache Cache in the Wild（ATC'25；arXiv 2506.02634 <https://arxiv.org/abs/2506.02634>） | GDFS 式、reuse-probability 逐出（C3） |
| Feng et al., AdaptCache <https://arxiv.org/abs/2509.00105>／EvicPress <https://arxiv.org/abs/2512.14946> | MCKP 與 marginal utility per byte（M2、N5） |
| Chen et al., LCR/LARU <https://arxiv.org/abs/2509.20979>；Guard, NeurIPS'25 <https://arxiv.org/abs/2507.16242> | robust learning-augmented LRU 已用於 LLM（K9） |
| Wu, Silwal, Zhang, ICLR 2026 <https://openreview.net/forum?id=R7fv5NWfMm> | KV prefix 逐出的 competitive 分析（J4） |
| Ganjihal, "When Prediction Doesn't Pay," 2026 <https://www.alphaxiv.org/abs/2608.llm-prefix-cache-recency-optimal>（arXiv 編號未確認） | recency ≈ OPT（hit ratio）（K11） |
| Tumkur et al., "Where Should the KV Cache Live?," arXiv 2609.16215 <https://arxiv.org/abs/2609.16215> | 摘要：收益主要來自層級容量比例而非放置策略；無 DROP、無量化、無 Belady（摘要核對） |
| Ganjihal, "Predictive Multi-Tier Memory Management for KV Cache," arXiv 2604.26968 <https://arxiv.org/abs/2604.26968> | Bayesian 重用預測驅動六層升降（摘要核對） |
| Ray, Feamster, Jiang, "An Internet for the KV Cache," arXiv 2608.01526 <https://arxiv.org/abs/2608.01526> | 以快取／CDN 觀點框定存 vs 重算（願景文） |
| Kim et al., arXiv 2411.07447 <https://arxiv.org/abs/2411.07447> | DBMS 啟發的 KV 替換策略（具體策略未核對） |
| Kanichai, De Matteis, Trivedi, py-kvcache, arXiv 2609.11744 <https://arxiv.org/abs/2609.11744> | vLLM＋NVMe 外部 KV 快取的效能刻畫（摘要未拆軟體 vs I/O） |

---

## §8 查證紀錄與未查證清單

**全文核對（下載 PDF／HTML 逐字看過相關段落）**：Atre et al. 2020；RAGCache；KVCache-in-the-Wild；LHD；Hyperbolic；Gill 2008；FOO；Dehghan et al.；PARROT；DTR；vLLM；Wilson et al. 1999；Nectar；Cao et al. 1995；Irani et al. 2003；InferCept；Continuum；CacheOPT；LRB；LIRS；SIEVE；S3-FIFO（摘要段）；CacheLib；Software-Defined Far Memory；Gray–Graefe 1997；Purohit et al. 2018；Kumar et al. CIDR 2025；PEVA 2025（首頁）；MatKV（HTML）；以及本地 KV 論文（Strata、Bidaw、Marconi、AdaptCache、EvicPress、Cake、bottlenecks2026、Tutti、ArkVale、OrbitFlow）。

**只核到摘要／出版頁**：其餘條目（見各條標註）。

**未查證／需注意**：
- Ganjihal 2026（K11）的標準 arXiv 編號：alphaXiv 頁顯示非標準 ID，未能確認。
- HRU 的 (1−1/e) 保證：來自搜尋摘要與課程投影片，原文 PDF 未逐字核對。
- Chrobak–Karloff–Payne–Vishwanathan 1991 的 min-cost flow 歸約：來自搜尋摘要，原文未逐字核對。
- Carra–Neglia–Michiardi 的正式 venue：未查證（引用 arXiv）。
- Harchol-Balter 教科書的 M/G/1 章節：未逐頁核對。
- TinyLFU 的 W-TinyLFU 變體：未在本文使用。
- 本地 InfiniGen 對 CacheLib／Capuchin、ArkVale 對 DTR 的引用語境：只確認在參考文獻中，未核對用途。
- EXPERIMENTS_20260919 表 2.2 的 16.3 µs：依表內數字驗算得約 13.0 µs（§3.2），需回原始 log 核對；「CPU/SSD 搬運 ÷3.77」是否為 INT4 實測（§3.1 第 4 點）需核對。
- Tiara 程式內部（是否計入 SSD 寫入量、Oracle 是否允許預取、block 大小）：本文未讀程式，只依 F1–F11 推測，相關敘述皆標為判讀。

**主要搜尋關鍵字（web）**：five-minute rule＋KV；delayed hits＋KV／learning-augmented；GreedyDual／GDSF＋KV；Landlord／hyperbolic／AdaptSize＋KV；LIRS／CLOCK-Pro／2Q／ARC＋KV；exclusive caching／DEMOTE＋KV；Checkmate／rematerialization＋KV；SHARDS／MRC＋KV；min-cost flow／FOO＋KV；Belady＋KV prefix；ski rental＋KV；Che approximation／TTL＋KV；Little's law＋KV；Amdahl＋KV offloading；learning-augmented＋KV；hit density／TinyLFU／admission＋KV SSD。**本地 grep**：以上所有經典名稱在 35 篇 KV 論文全文中的出現位置（見各條 (d)）。
