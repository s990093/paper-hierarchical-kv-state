# D8 卡片：TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding

- **連結**：https://arxiv.org/abs/2404.11912
- **venue／年份**：COLM 2024（arXiv comment）
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. 用原模型＋檢索出的稀疏 KV 當草稿模型，再用更小的模型草擬它〔摘要〕。
  2. 無損（推測解碼）〔摘要〕。
  3. 卸載設定（兩張 4090）0.108 s/token〔摘要〕。
- **和 D8 的關係**：方向 11（推測式還原）：推測解碼＋分層 KV 已有人做，但目標是 decode，不是還原期間。
