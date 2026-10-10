# D8 卡片：Tail-Optimized Caching for LLM Inference

- **連結**：https://arxiv.org/abs/2510.15152
- **venue／年份**：arXiv 2510.15152（2025-10）；venue 未查證
- **讀了哪裡**：arXiv 摘要頁（PDF 已下載到 d8_20261010，未細讀）
- **三行摘要**：
  1. LRU 對尾延遲可以任意差，因為它不看對話長度的差異〔摘要〕。
  2. Tail-Optimized LRU：兩行修改，優先淘汰不太會影響下一輪的條目，並證明在一個隨機模型下最佳〔摘要〕。
  3. WildChat 上 P90 TTFT 少最多 27.5%、P95 少 23.9%、200 ms SLO 違規少 38.9%〔摘要〕。
- **和 D8 的關係**：方向 13：SLO 導向淘汰已有理論與做法；D8 只能加「還原成本用 Cake 算」這一點。
