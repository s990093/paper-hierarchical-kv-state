# D6_Lynx Lynx: Progressive Speculative Quantization for accelerating KV Transfer in Long-Context Inference

- 出處：Wenchen Han（UCL）, Gingfung Matthew Yeung, Marco Barletta, William Toner, Amory Hoste 等（Huawei），arXiv 2607.01831v1（2026-07-02）。PDF 頁首的「SIGCOMM'18」是樣板佔位，venue 未查證。OpenAlex 搜尋找到的。PDF 在 `/mlsteam/data/tiara/papers_d6/arXiv2607.01831.pdf`。讀了摘要（arXiv abs 頁）和 p.6、p.8、p.10（grep 定位）。

- 精度決定：
  - **做什麼**：KV 先量化成 8-bit（含 α-law 對數轉換），再拆成兩條流：高 4 bit 是 Anchor 流（像指數，決定數量級），低 4 bit 是 Residual 流（像尾數，在區間內做線性內插）〔原文 p.6〕。
  - **何時**：傳輸時（P/D 分離，prefill 傳給 decode）。先收到 Anchor 就開始 speculative decode，Residual 同時在傳；Residual 收完後，用完整的 Q_full 做一次平行 forward 驗證 speculative 的 token：接受最長的正確前綴，改正第一個分歧處，保證輸出分布跟 Q_full 相同〔摘要；原文 p.8 §4.3〕。〔判讀：Q_full 應是合起來的 8-bit 碼，不是 BF16。〕
  - **粒度**：位元平面（每個值拆兩半），不是 token。
  - **用什麼資訊**：無，固定拆法。

- 有沒有和重算／載入並行結合？精度跟位置有關嗎？
  - 沒有重算。它並行的是「用 4-bit 先算」和「繼續傳剩下的 4 bit」。
  - 精度跟位置無關。

- 反量化／解碼成本：原文承認比單純 INT8／INT4 多了計算開銷，「通常不大」〔原文 p.10〕；我讀的段落沒有給具體數字。

- 品質與位置的關係：沒有量位置。TTFT 跟 INT4 相當，但傳的是 8-bit；某工作負載（MMLU + Qwen）上 TT1T 比 INT8 快 0.87 s〔原文 p.10〕。

- 有沒有和延後版比？沒有，但它的設計本身就是「不必先決定精度」。

- 和 D6 的關係〔判讀〕：
  - **繞開「寫入時精度不可逆」的另一種做法**：用位元平面的方式存，讀的時候可以先只讀高位（快），需要時再讀低位（精），而且不用像 CacheGen 那樣存好幾個版本。
  - 套到 Cake 上的想像：載入那段先讀 Anchor 就能讓 b 往前移，Residual 晚點補。這是 D6 可以考慮的「延後型」變體，但 Lynx 的上限是 8-bit，不是 BF16。
  - 這是 P/D 傳輸的論文，不是儲存，套到 SSD 要另外驗證。

- 證據等級：C+（摘要加三頁細讀，其餘未讀）。
