# C_KVBuffer KVBuffer: IO-aware Serving for Linear Attention

- **出處**：Longwei Zou, Lin Zhong（Yale）。arXiv 2605.19049v1，2026-05-18，cs.LG，頁首標「Preprint」。venue：未查證。<https://arxiv.org/abs/2605.19049>。讀了 p.1–2、p.7（PDF 實體頁；kernel 細節與附錄沒讀）。
- **寫入時做了什麼決定**：**線性注意力層存什麼格式**（GPU 記憶體內，不是分層儲存）。
  - decode 時先把最近 token 的 K、V 暫存起來，滿了再一次併入線性注意力狀態，而不是每步更新狀態〔原文 p.1 摘要、p.2〕。
  - 短 context（原文：「d 個或更少 token」，d 是 hidden dimension）時，只存每個 token 的 K、V，**根本不建立狀態**，直接從 KV 算輸出〔原文 p.2〕。
  - speculative decoding 時，暫存 draft token 的 KV，驗證後只用被接受的 token 更新一次狀態〔原文 p.2〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：KV 一旦併進狀態就拆不回來（狀態是有損的摘要）〔判讀〕。所以「先存 KV、晚點再併」是可以的，「先併、晚點要 KV」不行——方向和 N1 一致。
- **有沒有和延後版、寫穿版、背景版比較？**：比的是 GPU 上的三種計算形式：逐步遞迴（SGLang 現行）、chunkwise（KVBuffer）、只用 KV〔原文 p.1–2〕。decode 延遲最多降 45.17%；驗證 4 個 draft token 時可服務的請求數多 5 倍；端到端吞吐最多 1.46 倍〔原文 p.1–2〕。沒有任何分層儲存或延後寫入的比較。
- **硬體**：4 張 NVIDIA L40S〔原文 p.7〕。
- **模型架構、模態**：Qwen3-Next（Gated DeltaNet 混合），在 SGLang 上實作；文字〔原文 p.1〕。
- **原文給的大小**：Qwen3-Next 的一個 Gated DeltaNet 層，狀態約 **2 MB**，比每個 token 的 KV 大約兩個數量級；驗證 4 個 draft token 時每個請求多占 384 MB〔原文 p.1–2〕。
  - 和本文件的算術核對〔算術〕：config 的 linear_num_value_heads＝32、key／value head dim＝128，狀態＝32×128×128＝524,288 個元素；BF16 是 1 MiB、FP32 是 2 MiB。原文的 2 MB 對得上 **FP32 狀態**〔判讀：原文沒寫 dtype〕。
- **和本研究的關係**
  - **支持 H7 的前提**：一個線性層的狀態比一個 token 的 KV 大約兩個數量級〔原文 p.1〕，所以「每個 block 都存一份狀態檢查點」很貴，存哪些位置才值得決定。
  - **N4 的一個小變形**〔判讀〕：對很短的段落，「存最近 token 的線性層 KV」比「存一份狀態」小。如果某個檢查點離下一個檢查點很近，存中間 token 的線性層 KV（之後再併）可能比存兩份狀態便宜。原文只在 GPU 上做，沒有談跨請求或分層。
  - 不威脅任何 H（沒有更簡單的分層做法）。
- **證據等級**：〔原文 p.X〕；dtype 推斷與 N4 延伸〔判讀〕。
