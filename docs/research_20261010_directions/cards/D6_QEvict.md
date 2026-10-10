# D6_QEvict QEvict: Recoverable Quantized KV Eviction for Attention-Drift-Robust Long-Context Decoding

- 出處：Ayushman Garg, Akshita Gupta, Shaswata Bhattacharya, Abhishek Gupta, Sandeep Kumar, Manoj Kumar（IIT Roorkee、IIT Delhi），arXiv 2608.05326，venue 未查證。PDF 在 `/mlsteam/data/tiara/papers/08_QEvict_arXiv2608.05326.pdf`。讀了 p.1、p.4、p.6–7、p.19（grep 定位）。

- 精度決定：
  - **做什麼**：三層。高信心的 window 保持全精度；中間的 window 量化成可回收的低 bit 層（預設 INT2）；最低信心的才刪掉〔原文 p.1〕。
  - **何時**：decode 期間每 Ω 個生成 token（預設 Ω=8）做一次 routing，依累積 attention 分數升降級〔原文 p.6–7〕。
  - **粒度**：window（預設 8 個 token）。
  - **用什麼資訊**：累積 attention 分數。另外保護 sink（預設 5 個 token）和最近區域（預設 32 個 token），兩者都是全精度〔原文 p.4、p.7〕。

- 有沒有和重算／載入並行結合？精度跟位置有關嗎？
  - 沒有重算或載入，GPU 內方法。
  - **跟位置有關**：開頭的 sink 和最近區域固定全精度，中間依分數決定〔原文 p.4〕。

- 反量化／解碼成本（Table 4，Llama-3.1-8B，prefill 256、生成 1024、batch 32；硬體為 A100 80GB，見 p.19）：
  - FlashAttention-2 後端：TPOT 從 Full-KV 的 59.74 ms 變成 146.35 ms，throughput 從 535 掉到 218 tok/s；峰值記憶體 29.54 → 20.78 GB〔原文 p.7〕。原因是每步都要反量化並重組 cache，還要在 routing 時物化 attention 機率〔原文 p.7、p.19〕。
  - Eager/SDPA 後端：TPOT 84.05 → 76.26 ms（反而變快），TTFT 只多 0.5%〔原文 p.7〕。
  - 結論：反量化開銷的方向跟 attention 後端有關。

- 品質與位置的關係：保護 sink 和最近區域是設計前提，論文沒有單獨做「拿掉保護」的消融（我讀的段落裡沒看到）。

- 有沒有和延後版比？
  - 比的是「丟掉 vs 留低 bit」（可回收 vs 不可回收的逐出），不是「現在量化 vs 晚點量化」。

- 和 D6 的關係〔判讀〕：
  - **直接證明「寫入時精度不可逆」**：所謂「recoverable」只是不會被永久刪掉；升級回全精度層時，重建的是**同一個低 bit 近似**，只是換成執行用的 dtype。之後再降級會重用第一次降級的編碼，避免誤差累積〔原文 p.6〕。BF16 的資訊從第一次降級就沒了。這正是 D6 擔心的問題，延後版（存 BF16）不會有。
  - 它的位置政策（sink 和尾端全精度、中間低 bit）就是 D6 列的一種「中間低、兩頭高」的形狀，但它是在 GPU decode 內，不涉及 Cake 的 b。

- 證據等級：B。
