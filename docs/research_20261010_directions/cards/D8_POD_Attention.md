# D8 卡片：POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference

- **連結**：https://arxiv.org/abs/2410.18038
- **venue／年份**：ASPLOS 2025（arXiv comment）
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. prefill 吃算力、decode 吃記憶體頻寬；混合 batch 的 attention 仍各算各的〔摘要〕。
  2. 一個 kernel 讓 prefill 和 decode 的 attention 在同一個 SM 上並行〔摘要〕。
  3. attention 最多快 59%（平均 28%）〔摘要〕。
- **和 D8 的關係**：方向 4：decode 多的時候，重算（prefill 型工作）的邊際成本可能比單獨跑時低——這是它的硬體基礎。
