# D8 卡片：Llumnix: Dynamic Scheduling for Large Language Model Serving

- **連結**：https://arxiv.org/abs/2406.03243
- **venue／年份**：OSDI 2024（arXiv comment「To appear at OSDI '24」）
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. 在多個模型實例之間即時重新排程請求，像 OS 的 context switch〔摘要〕。
  2. 用 live migration 搬請求與它在記憶體裡的狀態（KV）〔摘要〕。
  3. 尾延遲改善一個數量級〔摘要〕。
- **和 D8 的關係**：方向 6：「搬 KV 而不是重算」在搶佔／遷移場景早有系統。
