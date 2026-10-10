# D6_KIVI KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

- 出處：Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen (Henry) Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu，ICML 2024（PMLR 235，見 p.1 頁尾）。https://arxiv.org/abs/2402.02750 。讀了 p.5–9 的方法、實驗、消融（grep 定位）；p.1–4 只掃過。

- 精度決定：
  - **做什麼**：Key 按 channel、Value 按 token 量化成 2-bit 或 4-bit，group size G=32〔原文 p.5–6〕。
  - **何時**：GPU 內、decode 過程中。最近的 R 個 token 先放在全精度的 residual 區；Key 的 residual 滿 R 個才一起量化，Value 用佇列，最舊的一個被擠出來時才量化〔原文 p.5〕。prefill 時傳給下一層的是精確的 KV，只有存起來的是量化版〔原文 p.5〕。
  - **粒度**：每 G 個 token 一組；全精度窗口 R=128〔原文 p.6〕。
  - **用什麼資訊**：只看位置（最近 R 個），不看內容。

- 有沒有和重算／載入並行結合？精度跟位置有關嗎？
  - 沒有重算或載入，純 GPU 內的方法。
  - **精度跟位置有關**：最近的 R 個 token 永遠是全精度。原文稱這是「local relevant tokens 的全精度滑動窗口」〔原文 p.6〕。

- 反量化／解碼成本：
  - 反量化跟矩陣乘法融合在一起（CUDA），分組量化用 Triton〔原文 p.6〕。**沒有單獨量反量化時間**。
  - 端到端：A100 80GB、Llama-2-7B，記憶體用量相近時，batch 可大 4 倍，throughput 高 2.35–3.47×〔原文 p.8〕。

- 品質與位置的關係：
  - **全精度窗口很重要**：Table 3，Llama-2-13B 在 GSM8K 上，16-bit 22.67；「全部都量化」的 fake 2-bit（K per-channel、V per-token）只有 12.21；KIVI-2（有 R=128 全精度窗口）20.77〔原文 p.8〕。原文說這個滑動窗口「對 GSM8K 這種難的生成任務是關鍵」〔原文 p.6、p.8〕。
  - 窗口長度本身：Table 5，R=32／64／96／128 時 GSM8K 分別 20.62／19.86／20.55／20.77，沒有一致趨勢；但「要有夠長的窗口」〔原文 p.9〕。

- 有沒有和延後版比？
  - 有一個 token 層級的類比：KIVI 的 residual 窗口本身就是「先不量化，滿了才量化」的延後做法；它跟「全部立刻量化」（fake quant）比，延後版大勝（上面的 12.21 vs 20.77）〔原文 p.8〕。但這是 GPU 內 decode 階段的延後，不是 Tiara 那種跨層級的延後。

- 和 D6 的關係〔判讀〕：
  - **對「尾端 BF16」的版本是支持**：靠近生成位置的 token 用高精度有實證。
  - **但要小心套用**：KIVI 的「最近」是 decode 時最新產生的 token。對存起來再還原的前綴來說，最靠近生成位置的是新的使用者問題（永遠是新算的 BF16），其次才是前綴的尾端。
  - **和 Cake 有衝突**：Cake 是從尾端開始載入，所以如果 D6 量化載入的那段，被量化的正好是前綴尾端，也就是 KIVI 認為最該保留高精度的區域（靠近問題）。這是 D6 位置政策要用實驗回答的問題，目前沒人量過（**NOT_MEASURED**）。

- 證據等級：B（方法、表 3、表 5、效率段細讀）。
