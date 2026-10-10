# D6_PatchKV PatchKV: Efficient KV Cache Recovery for Dynamically Edited LLM Contexts

- 出處：Guotao Yang, Rui Guo, Siwei He, Sheng Chen, Yitao Hu, Keqiu Li（天津大學），arXiv 2609.26219（2026-08-18），venue 未查證。WebSearch 找到的。PDF 在 `/mlsteam/data/tiara/papers_d6/arXiv2609.26219.pdf`。讀了 p.1–6、p.8–9（grep 定位後細讀 §IV-D、§V）。

- 精度決定：
  - **做什麼**：要載入的每個 block 選 FP16、K8/V8 或 K8/V4 其中一種〔原文 p.5〕。
  - **何時**：門檻和低 bit 品質測試在 calibration 資料上選好後凍結；每個 block 的精度標籤在 checkpoint offload 時決定並凍結〔原文 p.5–6〕。原文說 payload 建構、attention profiling、量化都在**resume 之前**、使用者思考或工具執行的空檔做；precision planning 4.44 ms、quantize-and-store 336.3 ms 都不算進 resume TTFT〔原文 p.6、p.9〕。
  - **粒度**：block；同精度的 block 會排成連續的 run〔原文 p.6〕。
  - **用什麼資訊**：存下來的 attention 質量 × 低 bit 重建殘差，算出每個 block 的風險 r_b（取 max 和 mean 的加權）〔原文 p.5〕。

- 有沒有和重算／載入並行結合？精度跟位置有關嗎？
  - **精度跟「會被重算還是被載入」有關**：block 先分成重算集 R（離編輯點近的「髒區」加上用 attention 挑的遠處 block）和傳輸集 F；**只有 F 會進入量化傳輸**〔原文 p.5〕。重算的部分不需要精度標籤。
  - 跟位置的關係是間接的：R 由「離編輯點的距離」決定，因為 KV 漂移會隨距離衰減〔原文 p.3、p.5〕；F 內的精度由 attention 風險決定，**不是由位置決定**。
  - 是否像 Cake 那樣讓重算和載入同時跑：有非同步 H2D 傳輸（CUDA streams、non-blocking copy）〔原文 p.6〕，但原文寫「兩組都放進目標 KV row 之後才開始 decode」〔原文 p.5〕，沒有描述兩頭往中間會合的排程。

- 反量化／解碼成本（A800、只用 CPU DRAM、H2D 20.4–23.3 GiB/s）〔原文 p.6〕：
  - Qwen3-32B、8K token checkpoint，**fetch + 反量化**時間：FP16 344.4 ms、INT8 183.6 ms、INT4 107.2 ms、PatchKV 混合 158.0 ms；CPU 記憶體 2,007.8／1,023.4／523.4／799.4 MiB〔原文 p.9〕。
  - 用一個融合 kernel 做 dequantize + RoPE 修正 + 寫進 KV page〔原文 p.6〕。

- 品質與位置的關係：
  - 固定重算集、只改精度時：PatchKV 比 FP16 快 15.0%、F1 掉 0.35；比 INT8 快 5.6%、F1 差 0.05 以內；INT4 比 PatchKV 再快 3.5–3.7%，但 F1 掉 1.15〔原文 p.8〕。
  - KV 漂移隨離編輯點的距離衰減（Fig. 3）〔原文 p.3〕；這是編輯場景特有的現象，不是一般的位置敏感度。

- 有沒有和延後版比？沒有。它的「offload 時決定、resume 前完成」介於寫入時和延後之間。

- 和 D6 的關係〔判讀〕：
  - **「精度跟重算／載入切分綁在一起」最接近的先例**：重算的部分不存精度，載入的部分 per-block 混合精度。
  - **但不是 D6 那種**：(1) 切分由編輯位置決定，不是 Cake 的成本會合點 b；(2) 載入段的精度看 attention 風險，不看位置或離 b 的距離；(3) 沒有兩頭並行。
  - **效果大小的參考**：在它的設定下，混合精度比均勻 INT8 只快 5.6%〔原文 p.8〕。D6 的位置政策即使成立，收益量級可能也類似。
  - 它的 8K、Qwen3-32B fetch + 反量化數字可以當 Tiara 量測的參照，但硬體不同，不能直接用（Tiara 自己的成本 **NOT_MEASURED**）。

- 證據等級：B。
