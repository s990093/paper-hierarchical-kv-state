# D6_AdaptCache AdaptCache: KV Cache Native Storage Hierarchy for Low-Delay and High-Quality Language Model Serving

- 出處：Shaoting Feng, Hanchen Li, Kuntai Du, … Junchen Jiang（芝加哥大學、Microsoft），arXiv 2509.00105v2（2026-01-15），只有 3 頁（像 workshop 短文；venue 未查證）。https://arxiv.org/abs/2509.00105 。p.1–3 全部讀過。

- 精度決定：
  - **做什麼**：每個 KV entry 選壓縮演算法（丟 token 或量化）、壓縮率、放 DRAM 還是 SSD〔原文 p.1–2〕。
  - **何時**：新 entry 產生時，對新 entry 和所有已存 entry 一起算「每省一單位空間掉多少 utility」，每一層貪婪地選「逐出／全精度存／壓縮」〔原文 p.2〕。所以**寫入時就決定**，而且會把舊 entry「壓得更兇」或逐出（降級時再決定）。
  - **粒度**：每個 KV entry（per context）。
  - **用什麼資訊**：Utility(i) = Freq(i)·(α·Quality − size/Bandwidth)。頻率用歷史命中次數估；品質—壓縮率曲線用 GPT-4o 產生的問題離線 profile；離線 profiler 也量了各裝置傳輸延遲和解壓開銷〔原文 p.2〕。

- 有沒有和重算／載入並行結合？精度跟位置有關嗎？
  - 沒有並行。「Prefill（重算）」只是 baseline〔原文 p.2〕。
  - 精度不依 token 位置。但動機段有一句跟位置有關：有些文字「只有開頭和結尾重要」，適合丟 token；有些在提供新資訊，量化可能較好〔原文 p.1〕。這是選演算法的理由，不是依位置分配精度。

- 反量化／解碼成本：profiler 有量解壓開銷，但論文**沒有給數字**〔原文 p.2〕。硬體為 1×A100、100 GB DRAM、400 GB SSD，磁碟讀取 1 GB/s〔原文 p.2〕。

- 品質與位置的關係：沒有量。

- 有沒有和延後版比？沒有。baseline 是「不壓縮 + DRAM/SSD」、「KIVI／StreamingLLM 固定壓縮率 + LRU」、「Prefill」〔原文 p.2〕。

- 和 D6 的關係〔判讀〕：
  - 它是 EvicPress 的前身，設計空間相同（per-context、utility、層與壓縮聯合決定），只是把量化（KIVI）放進選項。
  - 「把已存 entry 壓得更兇」代表它接受降級時的不可逆損失；論文沒討論是否保留原始副本（**未查證**，3 頁內沒寫）。
  - 跟位置、Cake 的 b 都無關。D6 的空白點不受它威脅。

- 證據等級：A（全文 3 頁讀完；但它本身只是初步結果）。
