# D8 卡片：Mitigating KV Cache Competition to Enhance User Experience in LLM Inference (CacheOPT)

- **連結**：https://arxiv.org/abs/2503.13773
- **venue／年份**：arXiv 2503.13773（2025-03）；venue 未查證
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. 預估輸出長度、預留 KV、主動配置，以減少搶佔〔摘要〕。
  2. 要搶佔時，挑 TBT SLO 寬、剩餘時間長、搶佔成本低的請求〔摘要〕。
  3. 搶佔方式在 swap 和重算之間「選延遲最短的」〔摘要〕——是二選一，不是兩者並行。
- **和 D8 的關係**：方向 6：Cake 式「兩者並行」載回被搶佔的請求，對照組就是 CacheOPT 的二選一。
