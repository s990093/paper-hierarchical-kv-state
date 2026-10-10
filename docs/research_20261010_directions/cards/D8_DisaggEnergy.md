# D8 卡片：Revisiting Disaggregated Large Language Model Serving for Performance and Energy Implications

- **連結**：https://arxiv.org/abs/2601.08833
- **venue／年份**：arXiv 2601.08833；venue 未查證
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. 比較 P/D 分離時不同 KV 傳輸路徑（不同記憶體與儲存層）的效能與能耗〔摘要〕。
  2. 用 DVFS 畫效能–能耗 Pareto 前緣〔摘要〕。
  3. 分離不一定帶來效能好處；分階段調頻也沒有省到能量〔摘要〕。
- **和 D8 的關係**：方向 1：有人量過「KV 走哪條路」的能耗，但沒有量「重算 vs 載入」的能耗比，也沒有混合還原。
