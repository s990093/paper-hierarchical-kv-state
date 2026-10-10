# D8 卡片：PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation

- **連結**：https://arxiv.org/abs/2509.19128
- **venue／年份**：arXiv 2509.19128；TMLR 2026（搜尋結果出現 mlanthology 頁，未開）→ venue 未查證
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. 非同步生成與訓練並行，用「生成中途換權重」（in-flight weight update）〔摘要〕。
  2. 生成只短暫暫停接收新權重，然後繼續生成進行中的序列〔摘要〕。
  3. 128 張 H100，學習速度約快 2 倍〔摘要〕。KV 是否保留：摘要沒寫；搜尋摘要說原文把舊 context 的 KV 視為過期並和重算版比較〔未讀原文，未查證〕。
- **和 D8 的關係**：方向 7：代表「沿用過期 KV」那一端。
