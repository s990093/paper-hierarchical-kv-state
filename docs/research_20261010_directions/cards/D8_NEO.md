# D8 卡片：NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference

- **連結**：https://arxiv.org/abs/2411.01142
- **venue／年份**：arXiv 2411.01142（2024-11）；venue 未查證
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. 把部分 attention 計算和 KV 搬到主機 CPU，讓 GPU 的 batch 變大〔摘要〕。
  2. 非對稱 GPU–CPU pipeline、依負載排程〔摘要〕。
  3. 吞吐：T4 最多 7.5 倍、A10G 26%、H100 14%〔摘要〕——GPU 記憶體越大，好處越小。
- **和 D8 的關係**：方向 14：在 192 GB 的 MI300X 上好處可能更小（依它自己的趨勢推論〔判讀〕）。
