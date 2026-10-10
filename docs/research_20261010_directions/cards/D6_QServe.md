# D6_QServe QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving

- 出處：Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, Song Han（MIT 等），MLSys 2025（p.1 頁尾）。https://arxiv.org/abs/2405.04532 （v3）。PDF 在 `/mlsteam/data/tiara/papers_d6/arXiv2405.04532.pdf`。讀了 p.1–2、p.4、p.6、p.8、p.10（grep 定位）。

- 精度決定：
  - **做什麼**：權重 4-bit、激活 8-bit、KV 4-bit（W4A8KV4）；用 SmoothAttention 壓 Key 的 outlier channel〔原文 p.1、p.6〕。
  - **何時**：KV 寫進 paged cache 時動態量化，每個 head 一組 scale／zero point，存在 KV page 裡、量化值的後面〔原文 p.8〕。
  - **粒度**：per-head，動態。
  - **用什麼資訊**：當下的 KV 數值；不看位置。

- 有沒有和重算／載入並行結合？精度跟位置有關嗎？沒有重算或載入，GPU 內方法。精度跟位置無關。

- 反量化／解碼成本：
  - **最有用的數字**：直接把 KV8 換成 KV4，在 L40S 上快 1.7×，但在 A100 上**反而慢 1.2×**。原因是 decode 的融合 attention kernel 在 A100 的 CUDA core 上變成算力瓶頸：天真地反量化一個 INT4 要 5 個 ALU 運算，而 A100 FP32 CUDA core 的 roofline 轉折點只有 9.8 Ops/Byte〔原文 p.10〕。
  - 改用 FP16 運算和 bit trick 把反量化降到每個元素 2 個運算後，比 TensorRT-LLM 的 KV8 kernel 快 1.5×（A100）〔原文 p.10〕。
  - Table 1（序列長度 128–1536，每次 attention）：KV8 0.09–0.62 ms，天真 KV4 0.10–0.69 ms，QServe KV4 0.07–0.41 ms〔原文 p.10〕。表裡天真 KV4 慢了約 0.86–0.90×，對應圖說中的 A100 情況〔判讀：表本身沒標 GPU〕。

- 品質與位置的關係：我讀的段落沒有位置相關結果。

- 有沒有和延後版比？沒有。

- 和 D6 的關係〔判讀〕：
  - **提醒反量化不是免費的**：低精度省下的位元組，在 GPU 上可能被反量化的 ALU 成本吃掉，而且跟硬體有關（A100 vs L40S 方向相反）。Tiara 有 RTX 3090 和 MI300X 兩個平台，INT4 的反量化成本可能完全不同，必須實測（**NOT_MEASURED**）。
  - 跟位置、Cake 的 b 無關。

- 證據等級：B（KV4 attention 與 KV cache 管理段落細讀）。
